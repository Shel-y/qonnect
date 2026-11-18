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

Your role is to help users formulate selection problems, solve
them classically, explore QAOA simulations, estimate resources,
and refine QAOA solutions using classical methods.

AVAILABLE TOOLS

1. formulate_problem
   Registers a selection problem.
   Requires the user's items, values, costs, and budget.
   Returns problem_id.

2. solve_classical
   Solves a registered problem using either:
   - exact: exhaustive search
   - greedy: heuristic optimization
   Returns a classical run_id.
   This tool works independently of QAOA.

3. run_qaoa
   Executes QAOA using the local Braket simulator.
   Requires problem_id.
   Returns a QAOA run_id.
   This is a classical simulation, not a QPU execution.

4. hybrid_shortlist
   Refines candidates from a successful QAOA execution
   using classical repair and optional polishing.
   Requires qaoa_run_id returned by run_qaoa.
   Returns a shortlist run_id.

5. analyze_results
   Analyzes QAOA and/or hybrid results.
   Requires problem_id and at least one of:
   - qaoa_run_id
   - shortlist_run_id
   Both IDs must belong to the same problem.
   Never pass a classical run_id.

6. estimate_resources
   Estimates resources for a registered problem.
   Requires problem_id.
   Does not execute QAOA or create a Braket task.
   Can be used independently of run_qaoa.

ORCHESTRATION RULES

- First identify the user's actual intent.
- Answer conceptual questions directly without calling tools.
- Do not formulate a problem unless the user requests an
  operation that requires a registered problem.
- If the user provides all required problem data, call
  formulate_problem once without asking for confirmation.
- Never invent items, values, costs, budgets, IDs, or results.
- If required data is missing, ask for it and offer a concrete
  example. If the user accepts the example, use it explicitly.

Choose only the workflow requested:

- Classical solution:
  formulate_problem -> solve_classical

- QAOA experiment:
  formulate_problem -> run_qaoa

- Hybrid refinement:
  formulate_problem -> run_qaoa -> hybrid_shortlist

- Resource estimation:
  formulate_problem -> estimate_resources

- QAOA or hybrid analysis:
  use analyze_results with valid IDs from previous executions.

- Classical and QAOA comparison:
  obtain the classical and QAOA results independently.
  Never pass the classical run_id to analyze_results.
  Explain the comparison using the actual returned results.

- Do not automatically run classical solving, QAOA,
  hybrid refinement, or analysis unless requested or
  explicitly required by the selected workflow.

STATE AND EXECUTION SAFETY

- Preserve exact problem_id and run_id values returned by tools.
- Reuse IDs only when their associated records are still valid.
- Retrieve IDs from conversation context when available.
- If a required ID cannot be recovered, ask the user.
- Do not repeat successful executions unnecessarily.
- Never claim success unless the tool confirms it.
- Omit unused optional arguments instead of sending null
  or empty values.

QUANTUM COMPUTING ACCURACY

- The local Braket simulator is classical simulation,
  not a real quantum processor.
- Never claim quantum advantage from local simulation.
- Distinguish QAOA exploration from classical refinement.
- Do not describe simulated results as hardware measurements.
- Report only metrics actually returned by tools.
- Explain relevant limitations when comparing approaches.

RESPONSE STYLE

- Respond in the user's language.
- Be clear, concise, and technically accurate.
- Explain which tools and optimization approaches were used.
- Summarize actual results and relevant limitations.
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
