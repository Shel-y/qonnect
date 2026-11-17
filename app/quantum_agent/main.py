"""
AgentCore entrypoint for the Strands quantum optimization agent.

Connects to the MCP server via SigV4-signed Streamable HTTP using
aws_iam_streamablehttp_client from mcp_proxy_for_aws, then runs the
Strands agent backed by Amazon Bedrock Nova 2 Lite.

Environment variables
---------------------
MCP_URL   : Required. Full URL of the MCP server endpoint,
            e.g. https://<agentcore-endpoint>/mcp
AWS_REGION: AWS region for Bedrock and SigV4 signing.
            Defaults to us-east-1.
"""

from __future__ import annotations

import os

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

# ============================================================
# CONFIGURACIÓN
# ============================================================

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID: str = "us.amazon.nova-2-lite-v1:0"

MEMORY_ID: str | None = os.environ.get("MEMORY_ID")
if not MEMORY_ID:
    raise EnvironmentError(
        "MEMORY_ID environment variable is required but not set."
    )

_MCP_URL: str | None = os.environ.get("MCP_URL")
if not _MCP_URL:
    raise EnvironmentError(
        "MCP_URL environment variable is required but not set. "
        "Set it to the full URL of the MCP server endpoint, "
        "e.g. https://<agentcore-endpoint>/mcp"
    )
MCP_URL: str = _MCP_URL


# ============================================================
# SYSTEM PROMPT  (copied verbatim from agent.py — do not alter)
# ============================================================

SYSTEM_PROMPT = """
You are an optimization assistant specialized in classical and
hybrid optimization using QAOA.

Your purpose is to help users formulate optimization problems,
solve them classically, explore quantum optimization with QAOA,
and compare the results.

You have access to six tools:

1. formulate_problem
   Registers and validates an optimization problem.
   Use this first when the user provides a new problem.
   It returns a problem_id that must be preserved and reused.

2. solve_classical
   Solves a registered problem using an exact classical method.
   Use this to establish a classical baseline.
   Preserve the returned run_id.

3. run_qaoa
   Executes QAOA for a registered problem.
   Use the problem_id returned by formulate_problem.
   Do not invent parameters or results.
   Use device="local" when running the local simulator.

4. hybrid_shortlist
   Performs the hybrid optimization workflow.
   Use the problem_id returned by formulate_problem.
   Preserve any run_id returned by this tool.

5. analyze_results
   Analyzes and compares existing optimization runs.
   Pass only valid run_ids returned by previous tools.
   Do not use a classical run_id where a QAOA or hybrid
   run_id is required.

6. estimate_resources
   Estimates resources for a registered optimization problem.
   Use the problem_id returned by formulate_problem.

ORCHESTRATION RULES

- Identify what the user is asking before selecting tools.
- Do not execute tools unless they are relevant to the request.
- For a new problem, call formulate_problem first.
- For a classical solution, call solve_classical after formulation.
- For a QAOA experiment, call run_qaoa after formulation.
- For a hybrid workflow, call hybrid_shortlist after formulation.
- For comparisons, use analyze_results with valid run_ids.
- For resource estimates, use estimate_resources.
- When a workflow requires multiple tools, execute them in the
  correct order and pass the IDs returned by previous tools.
- Never invent problem_ids, run_ids, parameters, measurements,
  execution times, costs, or optimization results.
- If a required ID is missing, retrieve it from the conversation
  context when possible. Otherwise, ask the user for clarification.
- Do not repeat a tool call if its result is already available
  and still valid.
- Do not claim a tool executed successfully unless it returned
  a successful result.

QUANTUM COMPUTING ACCURACY

- Clearly distinguish classical optimization from QAOA.
- The local simulator is a simulation, not a real quantum device.
- Never claim quantum advantage based on local simulation.
- Do not describe simulated results as hardware measurements.
- Explain limitations when comparing classical and quantum
  approaches.
- Report only metrics actually returned by the tools.

RESPONSE STYLE

- Respond in the user's language.
- Be clear, concise, and technically accurate.
- Explain which optimization approach was executed.
- Summarize relevant results and limitations.
- Do not expose internal reasoning.
- If the user sends an empty message, do nothing.
"""


# ============================================================
# AGENTCORE APP
# ============================================================

app = BedrockAgentCoreApp()


# ============================================================
# HELPERS
# ============================================================

_EXPOSED_TOOLS: frozenset[str] = frozenset({
    "formulate_problem",
    "solve_classical",
    "run_qaoa",
    "hybrid_shortlist",
    "analyze_results",
    "estimate_resources",
})


def _create_model() -> BedrockModel:
    session = boto3.Session(region_name=AWS_REGION)
    return BedrockModel(
        boto_session=session,
        model_id=MODEL_ID,
        temperature=0,
    )


def _mcp_client_factory():
    """Factory that creates a SigV4-signed Streamable HTTP MCP transport.

    Returns the async context manager expected by MCPClient's
    transport_callable parameter.
    """
    return aws_iam_streamablehttp_client(
        endpoint=MCP_URL,
        aws_service="bedrock-agentcore",
        aws_region=AWS_REGION,
    )


# ============================================================
# ENTRYPOINT
# ============================================================

@app.entrypoint
def handler(payload: dict) -> dict:
    """AgentCore invocation handler.

    Parameters
    ----------
    payload : dict
        Must contain non-empty string fields ``prompt``, ``actor_id``,
        and ``session_id``.

    Returns
    -------
    dict
        ``{"result": <agent response>}``
    """
    prompt: str | None = payload.get("prompt")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "payload must contain a non-empty 'prompt' string field."
        )

    actor_id = payload.get("actor_id")
    session_id = payload.get("session_id")
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("payload must contain a non-empty 'actor_id' string field.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("payload must contain a non-empty 'session_id' string field.")

    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        actor_id=actor_id.strip(),
        session_id=session_id.strip(),
    )
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=AWS_REGION,
    )

    model = _create_model()

    mcp_client = MCPClient(
        _mcp_client_factory,
        startup_timeout=30,
    )

    with mcp_client:
        all_tools = mcp_client.list_tools_sync()

        agent_tools = [
            tool for tool in all_tools
            if tool.tool_name in _EXPOSED_TOOLS
        ]

        loaded = {tool.tool_name for tool in agent_tools}
        missing = _EXPOSED_TOOLS - loaded
        if missing:
            raise RuntimeError(
                f"Required MCP tools are unavailable: {sorted(missing)}"
            )

        agent = Agent(
            model=model,
            tools=agent_tools,
            system_prompt=SYSTEM_PROMPT,
            session_manager=session_manager,
        )

        response = agent(prompt.strip())

    return {"result": response}


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":
    app.run()
    app.run()
