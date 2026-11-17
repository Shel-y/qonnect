# SPDX-License-Identifier: GPL-3.0-only
"""The six lab operations as plain functions (no Strands). JSON-compatible inputs and outputs.

Scope: ``SelectionProblem`` only (choose items maximizing value within an integer budget, avoiding conflicting
pairs, with optional mandatory items). No other problem family is supported yet.

Every operation returns a dict with ``ok`` True/False. Errors look like
``{"ok": False, "error": <code>, "message": <text>, "details": [...]}`` and never raise for bad input.
Results never use both ``status`` and ``content`` keys (Strands would read that as a raw ToolResult).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from pydantic import ValidationError

from strands_quantum_lab.analysis import HONESTY_NOTES, analyze
from strands_quantum_lab.classical import MAX_EXACT_FREE, exact_top_k, greedy_solve
from strands_quantum_lab.devices import classify_device, normalize_device
from strands_quantum_lab.hybrid import hybrid_shortlist as _hybrid_shortlist
from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qaoa import MAX_SIM_QUBITS, QuantumUnavailable, build_circuit, quantum_available
from strands_quantum_lab.qaoa import run_qaoa as _run_qaoa
from strands_quantum_lab.qubo import build_qubo, reduce, to_ising
from strands_quantum_lab.resources import estimate_resources as _estimate_resources
from strands_quantum_lab.store import LabStore, UnknownId, WrongRunKind

SUPPORTED_PROBLEM_TYPES = ("selection",)
SUPPORTED_DEVICES = ("local",)  # Braket LocalSimulator only; managed simulators / QPUs are not implemented
MAX_SAMPLES_RETURNED = 10


def _error(code: str, message: str, details: list[str] | None = None) -> dict:
    return {"ok": False, "error": code, "message": message, "details": details or []}


def _guard(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn lookup / quantum-availability errors into structured error results."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return fn(*args, **kwargs)
        except UnknownId as exc:
            return _error("unknown_id", str(exc.args[0]))
        except WrongRunKind as exc:
            return _error("wrong_run_kind", str(exc))
        except QuantumUnavailable as exc:
            return _error("quantum_unavailable", str(exc))

    return wrapper


def _selection(problem: SelectionProblem, x: list[int] | None) -> dict | None:
    if x is None:
        return None
    return {"selected": problem.selected_ids(x), "objective": problem.objective(x), "cost": problem.total_cost(x)}


def _check_device(device: str) -> dict | None:
    if device in SUPPORTED_DEVICES:
        return None
    if classify_device(device) == "paid":
        msg = f"device '{device}' (managed simulator or QPU) is not implemented in this version; nothing was run or billed"
        return _error("not_implemented", msg)
    return _error("device_not_supported", f"device '{device}' is not supported; this version only runs Braket LocalSimulator ('local')")


# 1 ---------------------------------------------------------------------------------------------------------------
def formulate_problem(
    store: LabStore,
    items: list[dict],
    budget: int,
    conflicts: list[list[str]] | None = None,
    name: str = "selection",
    description: str = "",
    problem_type: str = "selection",
) -> dict:
    """Validate a structured selection problem, build its QUBO and register it. The caller (e.g. an LLM) is
    responsible for turning natural language into ``items``/``budget``/``conflicts``; this function never parses text."""
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        return _error("unsupported_problem_type", f"problem_type '{problem_type}' is not supported; supported: {list(SUPPORTED_PROBLEM_TYPES)}")
    try:
        pairs = [tuple(c) for c in (conflicts or [])]
        if any(len(p) != 2 for p in pairs):
            return _error("invalid_problem", "each conflict must be a pair of two item ids")
        problem = SelectionProblem(name=name, items=items, budget=budget, conflicts=pairs)
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc']) or 'problem'}: {e['msg']}" for e in exc.errors()]
        return _error("invalid_problem", "the problem does not satisfy the schema", details)
    red = reduce(problem)
    qubo = build_qubo(red)
    pid = store.add_problem(problem, description)
    return {
        "ok": True,
        "problem_id": pid,
        "problem_type": "selection",
        "problem": problem.model_dump(mode="json"),
        "reduction": red.summary(),
        "qubo": {
            "variables": qubo.n_vars,
            "item_variables": qubo.n_items,
            "slack_variables": qubo.n_slack,
            "terms": len(qubo.terms),
            "penalty": qubo.penalty,
            "qaoa_qubits": qubo.n_vars,
            "within_local_simulation_limit": qubo.n_vars <= MAX_SIM_QUBITS,
        },
        "provenance": "classical",
    }


# 2 ---------------------------------------------------------------------------------------------------------------
@_guard
def solve_classical(store: LabStore, problem_id: str, method: str = "exact", top_k: int = 1) -> dict:
    """Classical reference solution: exact enumeration (top_k best) or a fast greedy heuristic."""
    sp = store.problem(problem_id)
    problem = sp.problem
    if method == "exact":
        if not 1 <= top_k <= 20:
            return _error("invalid_argument", "top_k must be between 1 and 20")
        try:
            cands, info = exact_top_k(problem, k=top_k)
        except ValueError as exc:
            return _error("too_large_for_exact", str(exc), [f"exact enumeration supports up to {MAX_EXACT_FREE} free items; use method='greedy'"])
    elif method == "greedy":
        cands, info = [greedy_solve(problem)], {}
    else:
        return _error("invalid_argument", "method must be 'exact' or 'greedy'")
    solutions = [_selection(problem, c.x) for c in cands]
    summary = {
        "method": method,
        "solutions": solutions,
        "info": info,
        "optimal": method == "exact",
        "provenance": "classical",
    }
    rid = store.add_run("classical", problem_id, {"method": method, "top_k": top_k}, cands, summary)
    return {"ok": True, "run_id": rid, "problem_id": problem_id, **summary}


# 3 ---------------------------------------------------------------------------------------------------------------
@_guard
def run_qaoa(
    store: LabStore,
    problem_id: str,
    layers: int = 1,
    shots: int = 1000,
    maxiter: int = 60,
    seed: int = 7,
    device: str = "local",
) -> dict:
    """QAOA with its classical variational loop, on Braket LocalSimulator."""
    device = normalize_device(device)
    if (err := _check_device(device)) is not None:
        return err
    if not 1 <= layers <= 3 or not 1 <= shots <= 100_000 or not 1 <= maxiter <= 500:
        return _error("invalid_argument", "layers must be 1-3, shots 1-100000, maxiter 1-500")
    sp = store.problem(problem_id)
    problem = sp.problem
    if not quantum_available():
        raise QuantumUnavailable("install the [braket] extra: amazon-braket-sdk, numpy, scipy")
    try:
        res = _run_qaoa(problem, layers=layers, shots=shots, maxiter=maxiter, seed=seed)
    except ValueError as exc:  # nothing left to optimize, or too many qubits
        return _error("not_runnable", str(exc))
    feasible_shots = sum(s.count for s in res.samples if s.feasible)
    best = res.best_feasible
    summary = {
        "device": device,
        "qubits": res.n_qubits,
        "layers": res.layers,
        "shots": res.shots,
        "gammas": res.gammas,
        "betas": res.betas,
        "expected_energy": res.expected_energy,
        "energy_history": res.energy_history,
        "circuit_evaluations": res.circuit_evaluations,
        "feasible_shot_fraction": round(feasible_shots / res.shots, 4),
        "distinct_assignments_measured": len(res.samples),
        "best_feasible_sample": None if best is None else {**_selection(problem, best.x), "shots": best.count},
        "top_samples": [
            {**_selection(problem, s.x), "shots": s.count, "feasible": s.feasible} for s in res.samples[:MAX_SAMPLES_RETURNED]
        ],
        "circuit": res.circuit,
        "timings_ms": res.timings_ms,
        "evaluation_mode": {"tuning": "simulator_exact_probabilities", "sampling": "braket_native_shots"},
        "provenance": "simulated",
        "notes": ["Braket LocalSimulator is a classical simulation of a quantum circuit, not a QPU.",
                  "Native Braket shots are not reproducible run to run."],
    }
    rid = store.add_run("qaoa", problem_id, {"layers": layers, "shots": shots, "maxiter": maxiter, "seed": seed, "device": device}, res, summary)
    return {"ok": True, "run_id": rid, "problem_id": problem_id, **summary}


# 4 ---------------------------------------------------------------------------------------------------------------
@_guard
def hybrid_shortlist(store: LabStore, qaoa_run_id: str, k: int = 5, polish: bool = True) -> dict:
    """QAOA proposes (its measured samples), classical repairs/polishes and decides among the top-k."""
    if not 1 <= k <= 20:
        return _error("invalid_argument", "k must be between 1 and 20")
    qrun = store.run(qaoa_run_id, kind="qaoa")
    problem = store.problem(qrun.problem_id).problem
    sl = _hybrid_shortlist(problem, qrun.result, k=k, polish=polish)
    entries = [
        {**_selection(problem, e.x), "origin": e.origin, "qaoa_shots": e.qaoa_count, "classical_actions": e.classical_actions}
        for e in sl.shortlist
    ]
    chosen = None if sl.chosen is None else {**_selection(problem, sl.chosen.x), "origin": sl.chosen.origin,
                                              "classical_actions": sl.chosen.classical_actions}
    summary = {
        "shortlist": entries,
        "chosen": chosen,
        "decision_rule": sl.decision_rule,
        "candidates_examined": sl.candidates_examined,
        "entries_needing_classical_help": sum(1 for e in sl.shortlist if e.origin != "qaoa"),
        "provenance": "simulated+classical",
    }
    rid = store.add_run("shortlist", qrun.problem_id, {"k": k, "polish": polish}, sl, summary, parent_run_id=qaoa_run_id)
    return {"ok": True, "run_id": rid, "problem_id": qrun.problem_id, "qaoa_run_id": qaoa_run_id, **summary}


# 5 ---------------------------------------------------------------------------------------------------------------
@_guard
def analyze_results(store: LabStore, problem_id: str, qaoa_run_id: str | None = None, shortlist_run_id: str | None = None) -> dict:
    """Explain a run: QAOA's own contribution, the classical refinement and the combined result, with the exact
    classical optimum only as a verification reference (when the problem is small enough to enumerate)."""
    problem = store.problem(problem_id).problem
    qrun = store.run(qaoa_run_id, kind="qaoa") if qaoa_run_id else None
    srun = store.run(shortlist_run_id, kind="shortlist") if shortlist_run_id else None
    for r in (qrun, srun):
        if r is not None and r.problem_id != problem_id:
            return _error("mismatched_runs", f"run '{r.id}' belongs to problem '{r.problem_id}', not '{problem_id}'")
    if srun is not None and qrun is None:
        qrun = store.run(srun.parent_run_id, kind="qaoa")
    if srun is not None and srun.parent_run_id != qrun.id:
        return _error("mismatched_runs", f"shortlist '{srun.id}' was built from '{srun.parent_run_id}', not '{qrun.id}'")
    if qrun is None:
        return _error("invalid_argument", "pass qaoa_run_id and/or shortlist_run_id")

    exact, info = [], {}
    if reduce(problem).n_free <= MAX_EXACT_FREE:
        exact, info = exact_top_k(problem, k=5)
    report = analyze(problem, exact, qrun.result, srun.result if srun else None, info) if exact else {}

    raw = qrun.result.best_feasible
    chosen = srun.result.chosen if srun else None
    contributions = {
        "qaoa": {
            "best_feasible_sample": None if raw is None else _selection(problem, raw.x),
            "feasible_shot_fraction": qrun.summary["feasible_shot_fraction"],
            "metrics": report.get("qaoa_alone"),
            "provenance": "simulated",
        },
        "classical_refinement": None if srun is None else {
            "chosen_origin": chosen.origin if chosen else None,
            "actions_on_chosen": chosen.classical_actions if chosen else [],
            "objective_gain_over_qaoa_best": None if (chosen is None or raw is None) else round(chosen.objective - raw.objective, 9),
            "entries_needing_classical_help": srun.summary["entries_needing_classical_help"],
            "provenance": "classical",
        },
        "combined": None if srun is None else {
            "chosen": None if chosen is None else _selection(problem, chosen.x),
            "constraints_satisfied": bool(chosen) and problem.is_feasible(chosen.x),
            "violations": [] if chosen is None else problem.violations(chosen.x),
            "metrics": report.get("hybrid"),
            "provenance": "simulated+classical",
        },
    }
    verification = None
    if exact:
        final = chosen.objective if chosen else (raw.objective if raw else None)
        verification = {
            "note": "Reference only: exact classical enumeration used to check results, not a competing method.",
            "optimal": _selection(problem, exact[0].x),
            "result_is_optimal": final is not None and abs(final - exact[0].objective) < 1e-9,
            "optimality_gap": None if final is None or not exact[0].objective else round((exact[0].objective - final) / exact[0].objective, 6),
            "provenance": "classical",
        }
    notes = list(HONESTY_NOTES) + ["There is no classical-to-QAOA feedback loop: QAOA explores once, then classical refines."]
    if srun is None:
        notes.append("No shortlist run was given: only QAOA's own contribution is reported.")
    elif raw is None:
        notes.append("QAOA measured no feasible selection; the result comes from classical repair of its samples.")
    elif chosen and chosen.objective > raw.objective:
        notes.append("The classical refinement improved on QAOA's best feasible sample.")
    elif chosen and chosen.origin == "qaoa":
        notes.append("The chosen selection was measured directly by QAOA and needed no classical change.")
    if not exact:
        notes.append("The problem is too large for exact verification; no optimality reference is given.")
    return {
        "ok": True,
        "problem_id": problem_id,
        "qaoa_run_id": qrun.id,
        "shortlist_run_id": srun.id if srun else None,
        "contributions": contributions,
        "verification": verification,
        "notes": notes,
    }


# 6 ---------------------------------------------------------------------------------------------------------------
@_guard
def estimate_resources(
    store: LabStore,
    problem_id: str,
    layers: int = 1,
    shots: int = 1000,
    circuit_evaluations: int = 60,
    device: str = "local",
    price_per_task_usd: float | None = None,
    price_per_shot_usd: float | None = None,
) -> dict:
    """Qubits, gates, depth and (only if current prices are given) potential QPU cost. Executes nothing."""
    if not 1 <= layers <= 3 or shots < 1 or circuit_evaluations < 1:
        return _error("invalid_argument", "layers must be 1-3; shots and circuit_evaluations must be positive")
    problem = store.problem(problem_id).problem
    est = _estimate_resources(problem, layers=layers, shots=shots, circuit_evaluations=circuit_evaluations,
                              price_per_task_usd=price_per_task_usd, price_per_shot_usd=price_per_shot_usd)
    depth = None
    if quantum_available() and est["qubits"]:
        qubo = build_qubo(reduce(problem))
        h, jj, _ = to_ising(qubo)
        circ = build_circuit(qubo.n_vars, h, jj, [0.1] * layers, [0.1] * layers, 1.0, measure_probabilities=False)
        depth = int(circ.depth)  # building a circuit object runs nothing
    return {
        "ok": True,
        "problem_id": problem_id,
        "device": device,
        **est,
        "depth": depth,
        "within_local_simulation_limit": est["qubits"] <= MAX_SIM_QUBITS,
        "executed_anything": False,
    }
