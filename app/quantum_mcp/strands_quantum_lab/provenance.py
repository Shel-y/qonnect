# SPDX-License-Identifier: GPL-3.0-only
"""Provenance computed from facts the plugin controls (tool name, its own store), never from the LLM. No Strands import.

Labels:
  classical  - computed on the local CPU by classical code.
  simulated  - a quantum circuit simulated on Braket LocalSimulator (classical hardware).
  real       - executed on quantum hardware. Only with evidence (a hardware task id); impossible in this version,
               where non-local devices are never executed.
  hybrid     - composite result; ``components`` keeps the label of each part (never collapsed into one label).
  none       - nothing was produced (blocked, failed or invalid input).
"""

from __future__ import annotations

from strands_quantum_lab.devices import classify_device
from strands_quantum_lab.store import LabStore, UnknownId

ASSIGNED_BY = "strands-quantum-lab:provenance-hook"
CLASSICAL_TOOLS = frozenset({"formulate_problem", "solve_classical", "estimate_resources"})
_REPORTED_EQUIVALENT = {"classical": "classical", "simulated": "simulated", "real": "real", "simulated+classical": "hybrid"}


def _base(label: str, components: dict, executed_on: str | None, evidence: str, reported: object) -> dict:
    expected = _REPORTED_EQUIVALENT.get(reported) if isinstance(reported, str) else None
    return {
        "label": label,
        "components": components,
        "executed_on": executed_on,
        "evidence": evidence,
        "reported_by_operation": reported,
        "consistent_with_operation": None if expected is None else expected == label,
        "assigned_by": ASSIGNED_BY,
    }


def _run_label(store: LabStore, run_id: str) -> str:
    run = store.run(run_id)
    if run.kind == "qaoa":
        return "simulated" if classify_device(run.params.get("device")) == "local" else "unverified"
    if run.kind == "shortlist":
        return "hybrid"
    return "classical"


def compute(tool: str, payload: dict | None, store: LabStore, outcome: str) -> dict:
    """``outcome`` is 'completed', 'failed' or 'blocked'; ``payload`` the tool's JSON result (None if not JSON)."""
    reported = payload.get("provenance") if isinstance(payload, dict) else None
    if outcome != "completed" or not isinstance(payload, dict) or not payload.get("ok"):
        return _base("none", {}, None, f"tool call {outcome}: no result was produced", reported)

    try:
        if tool in CLASSICAL_TOOLS:
            return _base("classical", {"computation": "classical"}, "local_cpu", f"{tool} runs only classical code", reported)

        if tool == "run_qaoa":
            run = store.run(payload["run_id"], kind="qaoa")
            device = run.params.get("device")
            if classify_device(device) == "local":
                return _base(
                    "simulated",
                    {"quantum_circuit": "simulated", "variational_optimizer": "classical"},
                    "braket_local_simulator",
                    f"run {run.id} stored with device '{device}' (Braket LocalSimulator)",
                    reported,
                )
            # Non-local devices are never executed in this version; without a hardware task id this is not 'real'.
            return _base("unverified", {"quantum_circuit": "unverified"}, None, f"device '{device}' without hardware evidence", reported)

        if tool == "hybrid_shortlist":
            run = store.run(payload["run_id"], kind="shortlist")
            parent = _run_label(store, run.parent_run_id)
            return _base(
                "hybrid",
                {"candidates": parent, "repair_and_polish": "classical", "decision": "classical"},
                "braket_local_simulator+local_cpu" if parent == "simulated" else None,
                f"shortlist {run.id} built from QAOA run {run.parent_run_id}",
                reported,
            )

        if tool == "analyze_results":
            inputs = {rid: _run_label(store, rid) for rid in (payload.get("qaoa_run_id"), payload.get("shortlist_run_id")) if rid}
            prov = _base("classical", {"analysis": "classical", "verification": "classical"}, "local_cpu",
                         "analysis of stored runs is classical code", reported)
            prov["inputs"] = inputs  # labels of the analyzed runs, taken from the store
            return prov
    except (UnknownId, KeyError, ValueError) as exc:
        return _base("none", {}, None, f"could not verify provenance: {exc}", reported)

    return _base("none", {}, None, f"unknown tool '{tool}'", reported)
