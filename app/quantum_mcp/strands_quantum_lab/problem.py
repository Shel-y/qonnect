# SPDX-License-Identifier: GPL-3.0-only
"""Structured optimization problem (PR-11, PR-12).

The LLM never writes QUBO coefficients. It fills in this schema; validation catches mistakes,
and a deterministic builder (``qubo.py``) turns it into a QUBO.

Family: selection with a budget and pairwise conflicts.
    maximize   sum(value_i * x_i)
    subject to sum(cost_i * x_i) <= budget
               x_i + x_j <= 1          for every conflict (i, j)
               x_i = 1                 for every mandatory item
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator

MAX_ITEMS = 24  # exact enumeration stays fast (2^24 subsets is the hard ceiling)


class Item(BaseModel):
    id: str = Field(min_length=1, description="Unique identifier of the item")
    value: float = Field(ge=0, description="Benefit of selecting the item (higher is better)")
    cost: int = Field(ge=0, description="Integer cost of the item, in the same unit as the budget (e.g. hours)")
    mandatory: bool = Field(default=False, description="The item must be selected")
    label: str | None = Field(default=None, description="Human-readable name")


class SelectionProblem(BaseModel):
    """Choose a subset of items that maximizes total value within a budget, avoiding conflicting pairs."""

    name: str = Field(default="selection", description="Short name of the problem")
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEMS)
    budget: int = Field(ge=0, description="Maximum total cost")
    conflicts: list[tuple[str, str]] = Field(default_factory=list, description="Pairs of item ids that cannot both be selected")

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [it.id for it in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("item ids must be unique")
        known = set(ids)
        for a, b in self.conflicts:
            if a not in known or b not in known:
                raise ValueError(f"conflict ({a}, {b}) references an unknown item")
            if a == b:
                raise ValueError(f"conflict ({a}, {b}) pairs an item with itself")
        mandatory = {it.id for it in self.items if it.mandatory}
        for a, b in self.conflicts:
            if a in mandatory and b in mandatory:
                raise ValueError(f"mandatory items {a} and {b} conflict: the problem is infeasible")
        if sum(it.cost for it in self.items if it.mandatory) > self.budget:
            raise ValueError("mandatory items exceed the budget: the problem is infeasible")
        return self

    # --- helpers working on bit vectors aligned with ``items`` ---

    @property
    def n(self) -> int:
        return len(self.items)

    def index(self) -> dict[str, int]:
        return {it.id: i for i, it in enumerate(self.items)}

    def conflict_pairs(self) -> list[tuple[int, int]]:
        idx = self.index()
        return sorted({tuple(sorted((idx[a], idx[b]))) for a, b in self.conflicts})

    def objective(self, x: Sequence[int]) -> float:
        return float(sum(it.value for it, xi in zip(self.items, x, strict=True) if xi))

    def total_cost(self, x: Sequence[int]) -> int:
        return sum(it.cost for it, xi in zip(self.items, x, strict=True) if xi)

    def violations(self, x: Sequence[int]) -> list[str]:
        out: list[str] = []
        cost = self.total_cost(x)
        if cost > self.budget:
            out.append(f"budget exceeded: {cost} > {self.budget}")
        for i, j in self.conflict_pairs():
            if x[i] and x[j]:
                out.append(f"conflict: {self.items[i].id} and {self.items[j].id}")
        for i, it in enumerate(self.items):
            if it.mandatory and not x[i]:
                out.append(f"mandatory item missing: {it.id}")
        return out

    def is_feasible(self, x: Sequence[int]) -> bool:
        return not self.violations(x)

    def selected_ids(self, x: Sequence[int]) -> list[str]:
        return [it.id for it, xi in zip(self.items, x, strict=True) if xi]
