
import boto3

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient


# ============================================================
# CONFIGURACIÓN
# ============================================================

AWS_PROFILE = "qonnect-demo"
AWS_REGION = "us-east-1"

MODEL_ID = "us.amazon.nova-2-lite-v1:0"

MCP_URL = "http://127.0.0.1:8000/mcp"


# ============================================================
# SYSTEM PROMPT
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
# MODELO
# ============================================================

def create_model():
    session = boto3.Session(
        profile_name=AWS_PROFILE,
        region_name=AWS_REGION,
    )

    return BedrockModel(
        boto_session=session,
        model_id=MODEL_ID,
        temperature=0,
    )


# ============================================================
# AGENTE Y SERVIDOR MCP
# ============================================================

def main():
    model = create_model()

    mcp_client = MCPClient(
        url=MCP_URL,
        startup_timeout=30,
    )

    with mcp_client:
        tools = mcp_client.list_tools_sync()

        exposed_tools = {
            "formulate_problem",
            "solve_classical",
            "run_qaoa",
            "hybrid_shortlist",
            "analyze_results",
            "estimate_resources",
        }

        agent_tools = [
            tool
            for tool in tools
            if tool.tool_name in exposed_tools
        ]

        loaded_tools = {
            tool.tool_name
            for tool in agent_tools
        }

        missing_tools = exposed_tools - loaded_tools

        if missing_tools:
            raise RuntimeError(
                f"Faltan herramientas MCP: {sorted(missing_tools)}"
            )

        agent = Agent(
            model=model,
            tools=agent_tools,
            system_prompt=SYSTEM_PROMPT,
        )

        print("\nAgente configurado correctamente.")
        print("Proveedor: Amazon Bedrock")
        print(f"Modelo: {MODEL_ID}")
        print(f"Región: {AWS_REGION}")
        print(
            "Herramientas activas:",
            sorted(loaded_tools),
        )

        while True:
            print(
                "\nTú (escribe FIN en una línea para enviar):"
            )

            lines = []

            while True:
                line = input()

                if line.strip().upper() == "FIN":
                    break

                if (
                    line.strip().lower()
                    in ("salir", "exit", "quit")
                    and not lines
                ):
                    return

                lines.append(line)

            prompt = "\n".join(lines).strip()

            if prompt:
                response = agent(prompt)
                print("\nAgente:", response)


if __name__ == "__main__":
    main()
