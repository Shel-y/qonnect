
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Callable, NotRequired, TypedDict

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from strands_quantum_lab import lab, provenance
from strands_quantum_lab.hooks import QubitLimit
from strands_quantum_lab.notebook import Notebook
from strands_quantum_lab.provenance import classify_device
from strands_quantum_lab.store import LabStore

class SelectionItem(TypedDict):
    id: str
    value: float
    cost: int
    mandatory: NotRequired[bool]
    label: NotRequired[str | None]

MAX_QUBITS = 16

# Shared state for this local MVP.
store = LabStore()
notebook = Notebook()
qubit_limit = QubitLimit(store, max_qubits=MAX_QUBITS)

server = MCPServer(
    name="strands-quantum-lab",
    version="0.1.0",
    instructions=(
        "Hybrid classical and quantum optimization laboratory. "
        "Formulate selection problems, solve them classically, "
        "run local QAOA simulations, refine candidates and analyze results. "
        "QAOA runs only on the free local simulator."
    ),
)


def _execute(
    tool_name: str,
    params: dict,
    operation: Callable[[], dict],
    blocked: bool = False,
) -> dict:
    """Execute, assign provenance and record one tool invocation."""
    started = time.perf_counter()
    tool_use_id = str(uuid.uuid4())
    print(f"\n[MCP EXEC] START | tool={tool_name} | id={tool_use_id}", flush=True)
    try:
        result = operation()
        if not isinstance(result, dict):
            result = {
                "ok": False,
                "error": "invalid_result",
                "message": "The laboratory returned an unexpected result.",
            }
    except Exception as exc:
        result = {
            "ok": False,
            "error": "internal_error",
            "message": str(exc),
        }

    duration = time.perf_counter() - started

    if blocked:
        outcome = "blocked"
    elif result.get("ok") is False:
        outcome = "failed"
    else:
        outcome = "completed"

    try:
        prov = provenance.compute(tool_name, result, store, outcome)
    except Exception as exc:
        print(
            f"[MCP EXEC] PROVENANCE ERROR | tool={tool_name} "
            f"| error={type(exc).__name__}: {exc}",
            flush=True,
        )
        prov = {
            "label": "none",
            "components": {},
            "executed_on": None,
            "evidence": "Provenance verification failed.",
            "reported_by_operation": result.get("provenance"),
            "consistent_with_operation": None,
            "assigned_by": "strands-quantum-lab:provenance-hook",
        }

    result = {**result, "provenance": prov}

    reason = None
    if outcome != "completed":
        reason = result.get("message") or result.get("error")

    try:
        notebook.record(
            tool_use_id=tool_use_id,
            tool=tool_name,
            status=outcome,
            params=params,
            payload=result,
            provenance=prov,
            duration_s=duration,
            reason=reason,
        )
    except Exception as exc:
        print(
            f"[MCP EXEC] NOTEBOOK ERROR | tool={tool_name} "
            f"| error={type(exc).__name__}: {exc}",
            flush=True,
        )
    print(
    f"[MCP EXEC] END | tool={tool_name} "
    f"| status={outcome} "
    f"| problem_id={result.get('problem_id')} "
    f"| run_id={result.get('run_id')} "
    f"| error={result.get('message') or result.get('error')} "
    f"| duration={duration:.3f}s",
    flush=True,
    )
    return result


@server.tool()
def formulate_problem(
    items: list[SelectionItem],
    budget: int,
    conflicts: list[list[str]] | None = None,
    name: str = "selection",
    description: str = "",
    problem_type: str = "selection",
) -> dict:
    """Register a new selection problem. Entry point for any new problem.

    WHEN: The user provides a problem to optimize.

    INPUT:
    - items: All items in one list. Each item has id (string), value (number),
    cost (integer >= 0), and optionally mandatory (boolean) and label (string).
    - budget: Maximum total cost (integer), provided by the user.
    - conflicts: Optional list of pairs of item IDs that cannot be selected together.
    - name, description: Omit unless provided by the user.
    - problem_type: Omit; the implementation defaults to "selection".

    BEHAVIOR: Extract structured data from the conversation. Never invent items,
    values, costs or budget. Set mandatory=true only when explicitly required.
    If all required data is present, call immediately without confirmation.
    Register all items in one call. Do not register the same problem twice.

    FLOW: Returns problem_id. Reuse this exact ID for solve_classical,
    run_qaoa and estimate_resources.
    """
    params = {
        "items": items,
        "budget": budget,
        "conflicts": conflicts,
        "name": name,
        "description": description,
        "problem_type": problem_type,
    }
    return _execute(
        "formulate_problem",
        params,
        lambda: lab.formulate_problem(
            store, items, budget, conflicts, name, description, problem_type
        ),
    )


@server.tool()
def solve_classical(
    problem_id: str,
    method: str = "greedy",
    top_k: int = 1,
) -> dict:
    """Solve a registered problem using a classical algorithm.

    WHEN: The user requests a solution or classical optimization.

    INPUT:
    - problem_id: Real ID returned by formulate_problem in this conversation.
    - method: "greedy" (default, fast heuristic) or "exact" (exhaustive
    search, exponential in the number of free items — only use when the user
    explicitly asks for the optimal or exact solution).
    - top_k: Number of solutions, range 1-20, only meaningful for exact.
    Omit unless requested.

    BEHAVIOR: Use "greedy" by default. Only switch to "exact" when the user
    explicitly asks for the optimal, exact, or best-possible solution.
    If no real problem_id is available, call formulate_problem first.
    Never send an empty or invented problem_id. This tool works independently
    of QAOA.

    FLOW: Returns a classical result and its run_id. Never pass this classical
    run_id to analyze_results.
    """
    params = {
        "problem_id": problem_id,
        "method": method,
        "top_k": top_k,
    }
    return _execute(
        "solve_classical",
        params,
        lambda: lab.solve_classical(store, problem_id, method, top_k),
    )


@server.tool()
def run_qaoa(
    problem_id: str,
    layers: int = 1,
    shots: int = 1000,
    maxiter: int = 25,
    seed: int = 7,
    device: str = "local",
) -> dict:
    """Execute QAOA using the local Braket simulator (classical simulation, not a QPU).

    WHEN: The user explicitly requests QAOA execution or quantum simulation.

    INPUT:
    - problem_id: Exact ID returned by formulate_problem.
    - layers: Range 1-3.
    - shots: Range 1-100000.
    - maxiter: Range 1-500. Default is 25 (fast exploration). Use higher
    values only when the user asks for a more thorough optimization.
    - seed: Random seed.
    - device: "local" only.

    SUCCESS AND OUTPUT:
    - Continue only if the response confirms success (ok=true) and contains
      a valid run_id.
    - Preserve that run_id exactly as qaoa_run_id for hybrid_shortlist
      or analyze_results.
    - If the response reports an error, or no valid run_id is returned,
      report the failure. Do not invent an ID or continue the QAOA workflow.

    BEHAVIOR:
    - Problems requiring more than 16 qubits are rejected.
    - This tool executes QAOA; it does not perform classical refinement.
    - Omit optional parameters unless the user specifies them.
    - For a QAOA-only request, do not call hybrid_shortlist or analyze_results.
    """
    params = {
        "problem_id": problem_id,
        "layers": layers,
        "shots": shots,
        "maxiter": maxiter,
        "seed": seed,
        "device": device,
    }

    def operation() -> dict:
        if classify_device(device) != "local":
            return {
                "ok": False,
                "error": "cost_guard_blocked",
                "message": (
                    "Only local simulation is allowed. "
                    "Paid and unknown devices are blocked."
                ),
            }

        required = qubit_limit.required_qubits(problem_id)
        if required > MAX_QUBITS:
            return {
                "ok": False,
                "error": "qubit_limit_exceeded",
                "message": (
                    f"Problem requires {required} qubits; "
                    f"the configured limit is {MAX_QUBITS}."
                ),
                "required_qubits": required,
                "max_qubits": MAX_QUBITS,
            }

        return lab.run_qaoa(
            store, problem_id, layers, shots, maxiter, seed, device
        )

    # Non-local devices are blocked before the laboratory is called.
    is_blocked = classify_device(device) != "local"

    return _execute("run_qaoa", params, operation, blocked=is_blocked)


@server.tool()
def hybrid_shortlist(
    qaoa_run_id: str,
    k: int = 5,
    polish: bool = True,
) -> dict:
    """Refine QAOA candidates using classical repair and optional polishing.

    WHEN: The user requests hybrid optimization or refinement of QAOA results.

    INPUT:
    - qaoa_run_id: Exact run_id returned by run_qaoa.
    - k: Number of candidates to retain, range 1-20.
    - polish: Whether to apply classical polishing.

    BEHAVIOR: This tool does not register a problem or execute QAOA.
    It requires a successful run_qaoa. If run_qaoa returned an error,
    report it instead of calling this tool.

    FLOW: Returns the field "run_id". Pass this exact value as
    shortlist_run_id to analyze_results. This is classical refinement
    of QAOA candidates, not an independent quantum execution.
    """
    params = {
        "qaoa_run_id": qaoa_run_id,
        "k": k,
        "polish": polish,
    }
    return _execute(
        "hybrid_shortlist",
        params,
        lambda: lab.hybrid_shortlist(store, qaoa_run_id, k, polish),
    )


@server.tool()
def analyze_results(
    problem_id: str,
    qaoa_run_id: str | None = None,
    shortlist_run_id: str | None = None,
) -> dict:
    """Analyze QAOA and hybrid optimization results.

    WHEN: The user requests interpretation, evaluation or comparison
    of existing QAOA or hybrid executions.

    INPUT:
    - problem_id: Exact ID returned by formulate_problem.
    - qaoa_run_id: Optional run_id returned by run_qaoa.
    - shortlist_run_id: Optional run_id returned by hybrid_shortlist.

    VALID FLOWS:
    - QAOA only: problem_id + qaoa_run_id.
    - Hybrid only: problem_id + shortlist_run_id.
    - Comparison: problem_id + both IDs from the same problem.

    BEHAVIOR: At least one execution ID is required. Never pass a classical
    run_id. Omit unused ID arguments; never send null or empty strings.
    This tool analyzes existing results; it does not execute other tools.
    """
    params = {
        "problem_id": problem_id,
        "qaoa_run_id": qaoa_run_id,
        "shortlist_run_id": shortlist_run_id,
    }
    return _execute(
        "analyze_results",
        params,
        lambda: lab.analyze_results(
            store, problem_id, qaoa_run_id, shortlist_run_id
        ),
    )


@server.tool()
def estimate_resources(
    problem_id: str,
    layers: int = 1,
    shots: int = 1000,
    circuit_evaluations: int = 25,
    device: str = "local",
    price_per_task_usd: float | None = None,
    price_per_shot_usd: float | None = None,
) -> dict:
    """Estimate resources for a QAOA execution without running it.

    WHEN: The user asks about qubits, circuit resources or estimated costs.

    INPUT:
    - problem_id: Exact ID returned by formulate_problem.
    - layers: QAOA circuit depth, range 1-3.
    - shots, circuit_evaluations: Positive integers; omit unless specified.
    Default circuit_evaluations (25) matches the default maxiter of run_qaoa.
    - device: Informational label; use "local". It does not select hardware.
    - price_per_task_usd, price_per_shot_usd: Optional actual prices
    provided by the user.

    BEHAVIOR: This tool estimates resources only. It does not execute QAOA,
    create a Braket task or return a run_id. Never invent prices.

    FLOW: Requires a real problem_id from formulate_problem. It can be used
    before or independently of run_qaoa.
    """
    params = {
        "problem_id": problem_id,
        "layers": layers,
        "shots": shots,
        "circuit_evaluations": circuit_evaluations,
        "device": device,
        "price_per_task_usd": price_per_task_usd,
        "price_per_shot_usd": price_per_shot_usd,
    }
    return _execute(
        "estimate_resources",
        params,
        lambda: lab.estimate_resources(
            store,
            problem_id,
            layers,
            shots,
            circuit_evaluations,
            device,
            price_per_task_usd,
            price_per_shot_usd,
        ),
    )


@server.custom_route("/ping", methods=["GET"])
async def ping(request: Request) -> JSONResponse:
    """Health check endpoint for AgentCore. Returns HTTP 200."""
    return JSONResponse({"status": "Healthy"})


async def main() -> None:
    port = int(os.environ.get("FASTMCP_PORT", "8000"))
    print("Starting Strands Quantum Lab MCP server...")
    print(f"Endpoint: http://0.0.0.0:{port}/mcp")
    await server.run_streamable_http_async(
        host="0.0.0.0",
        port=port,
        streamable_http_path="/mcp",
        stateless_http=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
