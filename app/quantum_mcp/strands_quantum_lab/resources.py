# SPDX-License-Identifier: GPL-3.0-only
"""Resource estimation without running anything.

Gate counts follow the circuit built in ``qaoa.build_circuit``. Prices are NOT hard-coded: Braket pricing
changes and differs per device, so a cost is computed only when the caller passes current prices.
"""

from __future__ import annotations

from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qubo import build_qubo, reduce, to_ising


def estimate_resources(
    problem: SelectionProblem,
    layers: int = 1,
    shots: int = 1000,
    circuit_evaluations: int = 60,
    price_per_task_usd: float | None = None,
    price_per_shot_usd: float | None = None,
) -> dict:
    red = reduce(problem)
    qubo = build_qubo(red)
    n = qubo.n_vars
    h, jj, _ = to_ising(qubo)
    rz = sum(1 for v in h if v)
    gates = n + layers * (len(jj) + rz + n)
    tasks = circuit_evaluations + 1  # variational loop + final measurement
    out: dict = {
        "reduction": red.summary(),
        "qubits": n,
        "item_qubits": qubo.n_items,
        "slack_qubits": qubo.n_slack,
        "two_qubit_terms": len(jj),
        "gates": gates,
        "layers": layers,
        "statevector_memory_mb": round((1 << n) * 16 / 1e6, 3) if n else 0.0,
        "on_hardware": {
            "tasks": tasks,
            "shots_total": tasks * shots,
            "note": "On a QPU every loop evaluation needs shots (no exact probabilities).",
        },
        "provenance": "classical",
    }
    if price_per_task_usd is not None and price_per_shot_usd is not None:
        out["on_hardware"]["estimated_cost_usd"] = round(tasks * (price_per_task_usd + shots * price_per_shot_usd), 4)
    else:
        out["on_hardware"]["estimated_cost_usd"] = None
        out["on_hardware"]["cost_note"] = "Pass current Amazon Braket prices for the target device to estimate cost."
    return out
