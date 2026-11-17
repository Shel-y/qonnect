# SPDX-License-Identifier: GPL-3.0-only
"""Classical solvers and polishing (pure Python, no extra dependencies)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qubo import Reduction, reduce

MAX_EXACT_FREE = 22


@dataclass
class Candidate:
    x: list[int]
    objective: float
    cost: int

    def key(self) -> tuple:
        # Higher objective first, then lower cost, then deterministic bit order.
        return (-round(self.objective, 9), self.cost, tuple(-b for b in self.x))


def _candidate(problem: SelectionProblem, x: list[int]) -> Candidate:
    return Candidate(x=list(x), objective=problem.objective(x), cost=problem.total_cost(x))


def exact_top_k(problem: SelectionProblem, k: int = 1, red: Reduction | None = None) -> tuple[list[Candidate], dict]:
    """The k best feasible selections by exhaustive enumeration of the free items."""
    red = red or reduce(problem)
    n = red.n_free
    if n > MAX_EXACT_FREE:
        raise ValueError(f"exact enumeration limited to {MAX_EXACT_FREE} free items (got {n}); use greedy_solve")
    t0 = time.perf_counter()
    kept: list[Candidate] = []
    evaluated = 0
    for mask in range(1 << n):
        x = red.lift([(mask >> b) & 1 for b in range(n)])
        evaluated += 1
        if problem.is_feasible(x):
            kept.append(_candidate(problem, x))
            if len(kept) >= 4 * k + 256:  # bounded memory: keep only the best k so far
                kept = sorted(kept, key=Candidate.key)[:k]
    best = sorted(kept, key=Candidate.key)[:k]
    return best, {"subsets_evaluated": evaluated, "free_items": n, "solve_ms": round((time.perf_counter() - t0) * 1000, 3)}


def greedy_solve(problem: SelectionProblem) -> Candidate:
    """Fast heuristic: mandatory items first, then by value/cost ratio, skipping conflicts and over-budget items."""
    x = [1 if it.mandatory else 0 for it in problem.items]
    return _candidate(problem, fill(problem, x))


def fill(problem: SelectionProblem, x: list[int]) -> list[int]:
    """Greedily add items that keep the selection feasible, best value/cost ratio first."""
    x = list(x)
    pairs = problem.conflict_pairs()
    order = sorted(range(problem.n), key=lambda i: -problem.items[i].value / max(problem.items[i].cost, 1e-9))
    cost = problem.total_cost(x)
    for i in order:
        it = problem.items[i]
        if x[i] or it.value == 0 or cost + it.cost > problem.budget:
            continue
        if any((a == i and x[b]) or (b == i and x[a]) for a, b in pairs):
            continue
        x[i] = 1
        cost += it.cost
    return x


def repair(problem: SelectionProblem, x: list[int]) -> tuple[list[int], list[str]]:
    """Make a selection feasible: add mandatory items, then drop the worst offenders. Returns (x, actions)."""
    x = list(x)
    actions: list[str] = []
    for i, it in enumerate(problem.items):
        if it.mandatory and not x[i]:
            x[i] = 1
            actions.append(f"added mandatory {it.id}")
    for a, b in problem.conflict_pairs():
        if x[a] and x[b]:
            drop = min((a, b), key=lambda i: (problem.items[i].mandatory, problem.items[i].value))
            x[drop] = 0
            actions.append(f"dropped {problem.items[drop].id} (conflict)")
    while problem.total_cost(x) > problem.budget:
        sel = [i for i in range(problem.n) if x[i] and not problem.items[i].mandatory]
        drop = min(sel, key=lambda i: problem.items[i].value / max(problem.items[i].cost, 1e-9))
        x[drop] = 0
        actions.append(f"dropped {problem.items[drop].id} (budget)")
    return x, actions


def local_search(problem: SelectionProblem, x: list[int], max_rounds: int = 50) -> tuple[list[int], list[str]]:
    """First-improvement search over 1-flips and swaps, staying feasible. Returns (x, actions)."""
    x = list(x)
    actions: list[str] = []
    best = problem.objective(x)
    for _ in range(max_rounds):
        improved = False
        moves: list[tuple[list[int], str]] = []
        for i in range(problem.n):
            y = list(x)
            y[i] ^= 1
            moves.append((y, f"{'added' if y[i] else 'removed'} {problem.items[i].id}"))
        for i in range(problem.n):
            for j in range(problem.n):
                if x[i] and not x[j]:
                    y = list(x)
                    y[i], y[j] = 0, 1
                    moves.append((y, f"swapped {problem.items[i].id} -> {problem.items[j].id}"))
        for y, label in moves:
            if problem.is_feasible(y) and problem.objective(y) > best + 1e-9:
                x, best = y, problem.objective(y)
                actions.append(label)
                improved = True
                break
        if not improved:
            break
    return x, actions
