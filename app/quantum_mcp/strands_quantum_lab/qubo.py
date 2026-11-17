# SPDX-License-Identifier: GPL-3.0-only
"""Classical reduction + deterministic QUBO builder (adapted from Qonnect, LA-10).

Reduction (classical, before any quantum step):
  * mandatory items are fixed to 1 and their cost is taken from the budget;
  * items that conflict with a mandatory item, cost more than the remaining budget,
    or have zero value are fixed to 0;
  * if the remaining free items all fit in the budget, no slack qubits are needed.

QUBO over the free items x and slack bits s (minimize):
    -sum v_i x_i
    + P * (sum c_i x_i + sum a_j s_j - B')^2      only if the free items can exceed B'
    + P * sum_{(i,j) in conflicts} x_i x_j
Slack coefficients a_j = 1, 2, 4, ..., and a last one chosen so that the slack covers exactly 0..B'.
P = 1 + sum v_i, so any violation costs more than the whole achievable gain.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from strands_quantum_lab.problem import SelectionProblem


@dataclass
class Reduction:
    problem: SelectionProblem
    free: list[int]  # indices into problem.items
    fixed: dict[int, int]  # index -> 0/1
    residual_budget: int
    reasons: dict[str, str] = field(default_factory=dict)  # item id -> why it was fixed

    @property
    def n_free(self) -> int:
        return len(self.free)

    def lift(self, sub_x: list[int]) -> list[int]:
        """Full bit vector from an assignment of the free items."""
        x = [0] * self.problem.n
        for i, v in self.fixed.items():
            x[i] = v
        for pos, i in enumerate(self.free):
            x[i] = int(sub_x[pos])
        return x

    def summary(self) -> dict:
        return {
            "items": self.problem.n,
            "free_items": self.n_free,
            "fixed_to_1": [self.problem.items[i].id for i, v in self.fixed.items() if v == 1],
            "fixed_to_0": [self.problem.items[i].id for i, v in self.fixed.items() if v == 0],
            "residual_budget": self.residual_budget,
            "reasons": self.reasons,
        }


def reduce(problem: SelectionProblem) -> Reduction:
    fixed: dict[int, int] = {}
    reasons: dict[str, str] = {}
    for i, it in enumerate(problem.items):
        if it.mandatory:
            fixed[i] = 1
            reasons[it.id] = "mandatory"
    budget = problem.budget - sum(problem.items[i].cost for i in fixed)
    for i, j in problem.conflict_pairs():
        for a, b in ((i, j), (j, i)):
            if fixed.get(a) == 1 and b not in fixed:
                fixed[b] = 0
                reasons[problem.items[b].id] = f"conflicts with mandatory {problem.items[a].id}"
    for i, it in enumerate(problem.items):
        if i in fixed:
            continue
        if it.cost > budget:
            fixed[i] = 0
            reasons[it.id] = f"cost {it.cost} > remaining budget {budget}"
        elif it.value == 0:
            fixed[i] = 0
            reasons[it.id] = "zero value"
    free = [i for i in range(problem.n) if i not in fixed]
    return Reduction(problem=problem, free=free, fixed=fixed, residual_budget=budget, reasons=reasons)


@dataclass
class Qubo:
    n_items: int  # free items (first variables)
    slack_coeffs: list[int]
    terms: dict[tuple[int, int], float]  # (i, j) with i <= j; (i, i) is linear
    offset: float
    penalty: float

    @property
    def n_slack(self) -> int:
        return len(self.slack_coeffs)

    @property
    def n_vars(self) -> int:
        return self.n_items + self.n_slack

    def energy(self, bits) -> float:
        e = self.offset
        for (i, j), c in self.terms.items():
            if bits[i] and bits[j]:
                e += c
        return e


def slack_coefficients(bound: int) -> list[int]:
    """Coefficients whose 0/1 combinations cover exactly 0..bound."""
    if bound <= 0:
        return []
    k = bound.bit_length()
    coeffs = [1 << j for j in range(k - 1)]
    coeffs.append(bound - (sum(coeffs)))
    return coeffs


def _add_square(terms, offset: list[float], coeffs: dict[int, float], const: float, weight: float) -> None:
    """Add weight * (sum_v a_v v + const)^2 for binary v (v^2 = v)."""
    items = sorted(coeffs.items())
    for idx, (v, a) in enumerate(items):
        terms[(v, v)] += weight * (a * a + 2 * const * a)
        for w, b in items[idx + 1 :]:
            terms[(v, w)] += weight * 2 * a * b
    offset[0] += weight * const * const


def build_qubo(red: Reduction) -> Qubo:
    p = red.problem
    free_items = [p.items[i] for i in red.free]
    n = len(free_items)
    penalty = 1.0 + sum(it.value for it in free_items)
    terms: dict[tuple[int, int], float] = defaultdict(float)
    offset = [0.0]
    for k, it in enumerate(free_items):
        terms[(k, k)] -= it.value

    slack: list[int] = []
    if sum(it.cost for it in free_items) > red.residual_budget:
        slack = slack_coefficients(red.residual_budget)
        coeffs: dict[int, float] = {k: float(it.cost) for k, it in enumerate(free_items) if it.cost}
        for j, a in enumerate(slack):
            coeffs[n + j] = float(a)
        _add_square(terms, offset, coeffs, -float(red.residual_budget), penalty)

    pos = {i: k for k, i in enumerate(red.free)}
    for i, j in p.conflict_pairs():
        if i in pos and j in pos:
            a, b = sorted((pos[i], pos[j]))
            terms[(a, b)] += penalty

    clean = {k: v for k, v in terms.items() if abs(v) > 1e-12}
    return Qubo(n_items=n, slack_coeffs=slack, terms=clean, offset=offset[0], penalty=penalty)


def to_ising(qubo: Qubo) -> tuple[list[float], dict[tuple[int, int], float], float]:
    """QUBO -> Ising with x = (1 - z) / 2. Returns (h, J, const)."""
    h = [0.0] * qubo.n_vars
    jj: dict[tuple[int, int], float] = {}
    const = qubo.offset
    for (i, j), c in qubo.terms.items():
        if i == j:
            const += c / 2
            h[i] -= c / 2
        else:
            const += c / 4
            h[i] -= c / 4
            h[j] -= c / 4
            jj[(i, j)] = jj.get((i, j), 0.0) + c / 4
    return h, jj, const


def energy_vector(qubo: Qubo):
    """Energies of all 2^N assignments, qubit 0 = most significant bit (Braket's probability order)."""
    import numpy as np

    n = qubo.n_vars
    idx = np.arange(1 << n, dtype=np.int64)
    bits = [((idx >> (n - 1 - q)) & 1).astype(np.float64) for q in range(n)]
    e = np.full(1 << n, qubo.offset, dtype=np.float64)
    for (i, j), c in qubo.terms.items():
        e += c * (bits[i] if i == j else bits[i] * bits[j])
    return e


def index_to_bits(index: int, n_vars: int) -> list[int]:
    return [(index >> (n_vars - 1 - q)) & 1 for q in range(n_vars)]
