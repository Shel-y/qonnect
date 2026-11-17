# SPDX-License-Identifier: GPL-3.0-only
"""Hybrid shortlist: QAOA proposes, classical decides.

1. QAOA (simulated) is measured; its distinct item assignments are candidates, most frequent first.
2. Classical repair makes infeasible candidates feasible; local search adds polished variants (originals are kept).
3. The k best distinct candidates form the shortlist.
4. A classical decision rule picks one. The default is highest value, then lowest cost; callers may pass
   ``decide`` to apply criteria that do not fit in a QUBO (e.g. spread of deadlines, preferences).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from strands_quantum_lab.classical import Candidate, local_search, repair
from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qaoa import QaoaResult


@dataclass
class ShortlistEntry:
    x: list[int]
    objective: float
    cost: int
    qaoa_count: int  # shots that measured the originating assignment
    origin: str  # "qaoa" (feasible as measured) or "qaoa+classical" (repaired and/or polished)
    classical_actions: list[str]


@dataclass
class ShortlistResult:
    shortlist: list[ShortlistEntry]
    chosen: ShortlistEntry | None
    decision_rule: str
    candidates_examined: int


def default_decide(problem: SelectionProblem, shortlist: list[ShortlistEntry]) -> ShortlistEntry:
    return min(shortlist, key=lambda e: Candidate(e.x, e.objective, e.cost).key())


def hybrid_shortlist(
    problem: SelectionProblem,
    qaoa: QaoaResult,
    k: int = 5,
    max_candidates: int | None = None,
    polish: bool = True,
    decide: Callable[[SelectionProblem, list[ShortlistEntry]], ShortlistEntry] | None = None,
    decision_rule: str = "highest value, then lowest cost",
) -> ShortlistResult:
    seen: dict[tuple[int, ...], ShortlistEntry] = {}

    def keep(x: list[int], count: int, actions: list[str]) -> None:
        entry = ShortlistEntry(
            x=x,
            objective=problem.objective(x),
            cost=problem.total_cost(x),
            qaoa_count=count,
            origin="qaoa" if not actions else "qaoa+classical",
            classical_actions=actions,
        )
        key = tuple(x)
        if key not in seen or (seen[key].origin != "qaoa" and entry.origin == "qaoa"):
            seen[key] = entry

    examined = 0
    for s in qaoa.samples[:max_candidates]:
        examined += 1
        # Keep what QAOA measured (if feasible) AND the classically improved version, so polishing
        # adds candidates instead of collapsing the shortlist onto a few local optima.
        if s.feasible:
            keep(list(s.x), s.count, [])
            x, actions = list(s.x), []
        else:
            x, actions = repair(problem, s.x)
            keep(x, s.count, actions)
        if polish:
            y, more = local_search(problem, x)
            if more:
                keep(y, s.count, actions + more)
    ranked = sorted(seen.values(), key=lambda e: Candidate(e.x, e.objective, e.cost).key())[:k]
    chooser = decide or default_decide
    chosen = chooser(problem, ranked) if ranked else None
    return ShortlistResult(shortlist=ranked, chosen=chosen, decision_rule=decision_rule, candidates_examined=examined)
