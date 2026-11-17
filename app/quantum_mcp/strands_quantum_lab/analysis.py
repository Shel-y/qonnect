# SPDX-License-Identifier: GPL-3.0-only
"""Honest analysis of a hybrid experiment (LA-05).

Separates what QAOA delivered on its own (raw measured samples) from what the classical stages added
(repair, polishing, decision), and always reports against the exact classical reference.
"""

from __future__ import annotations

from strands_quantum_lab.classical import Candidate
from strands_quantum_lab.hybrid import ShortlistResult
from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qaoa import QaoaResult

HONESTY_NOTES = [
    "LocalSimulator is a classical simulation of a quantum circuit; it is not a QPU and shows no quantum advantage.",
    "At this size the exact classical solver is the reference: it is optimal by construction.",
]


def _ratio(value: float | None, best: float) -> float | None:
    if value is None:
        return None
    return round(value / best, 6) if best else (1.0 if value == best else None)


def analyze(
    problem: SelectionProblem,
    exact: list[Candidate],
    qaoa: QaoaResult | None = None,
    shortlist: ShortlistResult | None = None,
    exact_info: dict | None = None,
) -> dict:
    if not exact:
        return {"feasible": False, "notes": ["No feasible selection exists."] + HONESTY_NOTES}
    best = exact[0]
    top_keys = {tuple(c.x) for c in exact}
    k = len(exact)
    report: dict = {
        "problem": problem.name,
        "classical_reference": {
            "objective": best.objective,
            "cost": best.cost,
            "selected": problem.selected_ids(best.x),
            "top_k": k,
            **(exact_info or {}),
            "provenance": "classical",
        },
    }
    if qaoa is not None:
        feasible = [s for s in qaoa.samples if s.feasible]
        feasible_shots = sum(s.count for s in feasible)
        opt_shots = sum(s.count for s in qaoa.samples if tuple(s.x) == tuple(best.x))
        raw_best = qaoa.best_feasible
        report["qaoa_alone"] = {
            "qubits": qaoa.n_qubits,
            "layers": qaoa.layers,
            "shots": qaoa.shots,
            "feasible_shot_fraction": round(feasible_shots / qaoa.shots, 4),
            "optimum_shot_fraction": round(opt_shots / qaoa.shots, 4),
            "best_feasible_objective": raw_best.objective if raw_best else None,
            "approximation_ratio": _ratio(raw_best.objective if raw_best else None, best.objective),
            "recall_at_k": round(len(top_keys & {tuple(s.x) for s in feasible}) / k, 4),
            "distinct_assignments_measured": len(qaoa.samples),
            "timings_ms": qaoa.timings_ms,
            "provenance": "simulated",
        }
    if shortlist is not None and shortlist.chosen is not None:
        helped = sum(1 for e in shortlist.shortlist if e.origin != "qaoa")
        report["hybrid"] = {
            "chosen": problem.selected_ids(shortlist.chosen.x),
            "chosen_objective": shortlist.chosen.objective,
            "approximation_ratio": _ratio(shortlist.chosen.objective, best.objective),
            "recall_at_k": round(len(top_keys & {tuple(e.x) for e in shortlist.shortlist}) / k, 4),
            "shortlist_size": len(shortlist.shortlist),
            "entries_needing_classical_help": helped,
            "decision_rule": shortlist.decision_rule,
            "provenance": "simulated+classical",
        }
    report["notes"] = list(HONESTY_NOTES)
    if qaoa is not None and "hybrid" in report:
        alone = report["qaoa_alone"]["approximation_ratio"] or 0.0
        hybrid = report["hybrid"]["approximation_ratio"] or 0.0
        if hybrid > alone:
            report["notes"].append("The classical stages improved on QAOA's raw samples.")
        if report["hybrid"]["entries_needing_classical_help"]:
            report["notes"].append("Some shortlist entries only became good after classical repair or polishing.")
    return report
