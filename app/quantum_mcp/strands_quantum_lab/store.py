# SPDX-License-Identifier: GPL-3.0-only
"""In-memory registry so operations can be chained by id instead of resending JSON.

Problems get ids ``P1, P2, ...``; runs (classical, qaoa, shortlist) get ``R1, R2, ...``.
One store per plugin instance (and so per agent); nothing is persisted to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strands_quantum_lab.problem import SelectionProblem

RUN_KINDS = ("classical", "qaoa", "shortlist")


class UnknownId(KeyError):
    """The requested problem or run id does not exist in this store."""


class WrongRunKind(ValueError):
    """The run exists but is not of the kind the operation needs."""


@dataclass
class StoredProblem:
    id: str
    problem: SelectionProblem
    description: str = ""


@dataclass
class StoredRun:
    id: str
    kind: str
    problem_id: str
    params: dict
    result: Any  # engine object (QaoaResult, ShortlistResult, list[Candidate])
    summary: dict = field(default_factory=dict)  # JSON summary returned to the caller
    parent_run_id: str | None = None


class LabStore:
    def __init__(self) -> None:
        self._problems: dict[str, StoredProblem] = {}
        self._runs: dict[str, StoredRun] = {}
        self._p = 0
        self._r = 0

    def add_problem(self, problem: SelectionProblem, description: str = "") -> str:
        self._p += 1
        pid = f"P{self._p}"
        self._problems[pid] = StoredProblem(pid, problem, description)
        return pid

    def problem(self, problem_id: str) -> StoredProblem:
        try:
            return self._problems[problem_id]
        except KeyError:
            raise UnknownId(f"unknown problem_id '{problem_id}'; known: {sorted(self._problems)}") from None

    def add_run(self, kind: str, problem_id: str, params: dict, result: Any, summary: dict, parent_run_id: str | None = None) -> str:
        if kind not in RUN_KINDS:
            raise ValueError(f"unknown run kind {kind}")
        self.problem(problem_id)
        self._r += 1
        rid = f"R{self._r}"
        self._runs[rid] = StoredRun(rid, kind, problem_id, params, result, summary, parent_run_id)
        return rid

    def run(self, run_id: str, kind: str | None = None) -> StoredRun:
        try:
            run = self._runs[run_id]
        except KeyError:
            raise UnknownId(f"unknown run_id '{run_id}'; known: {sorted(self._runs)}") from None
        if kind is not None and run.kind != kind:
            raise WrongRunKind(f"run '{run_id}' is a '{run.kind}' run, but a '{kind}' run is required")
        return run

    def runs_for(self, problem_id: str, kind: str | None = None) -> list[StoredRun]:
        return [r for r in self._runs.values() if r.problem_id == problem_id and (kind is None or r.kind == kind)]
