# SPDX-License-Identifier: GPL-3.0-only
"""Hook logic. Phase B: CostGuard and QubitLimit (BeforeToolCallEvent). Phase C: ProvenanceHook and NotebookHook
(AfterToolCallEvent).

They run in the Strands event loop, outside the model: the LLM cannot skip or reconfigure them.
Approval is NOT authorization to execute: paid devices are not implemented, so even an approved call
ends in ``not_implemented`` from ``lab.run_qaoa`` and nothing runs or is billed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

from strands_quantum_lab import provenance
from strands_quantum_lab.devices import classify_device, normalize_device
from strands_quantum_lab.notebook import Notebook
from strands_quantum_lab.qubo import build_qubo, reduce
from strands_quantum_lab.store import LabStore, UnknownId

EXECUTING_TOOLS = frozenset({"run_qaoa"})  # tools that would run a circuit on a device (estimate_resources runs nothing)
APPROVE = "approve"
COST_INTERRUPT = "strands-quantum-lab:cost-approval"


class CostGuard:
    """Blocks paid/QPU/unknown devices unless explicitly allowed; allowed paid devices require human approval."""

    def __init__(self, allowed_paid_devices: Iterable[str] = ()):
        self.allowed_paid_devices = frozenset(normalize_device(d) for d in allowed_paid_devices)
        bad = [d for d in self.allowed_paid_devices if classify_device(d) != "paid"]
        if bad:
            raise ValueError(f"allowed_paid_devices must be paid Braket devices, got {bad}")

    def check(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool or event.tool_use["name"] not in EXECUTING_TOOLS:
            return
        device = normalize_device((event.tool_use.get("input") or {}).get("device", "local"))
        kind = classify_device(device)
        if kind == "local":
            return
        if kind == "unknown":
            event.cancel_tool = f"blocked by cost guard: unknown device '{device}' (only 'local' runs in this version)"
            return
        if device not in self.allowed_paid_devices:
            event.cancel_tool = (
                f"blocked by cost guard: '{device}' is a paid device (managed simulator or QPU) and is not allowed "
                "by this plugin's configuration"
            )
            return
        response = event.interrupt(
            COST_INTERRUPT,
            reason={
                "tool": event.tool_use["name"],
                "device": device,
                "device_kind": "paid",
                "message": f"Running on '{device}' could incur AWS charges. Reply '{APPROVE}' to allow the call.",
                "note": "In this version paid devices are not implemented: an approved call returns not_implemented and runs nothing.",
            },
        )
        if response != APPROVE:
            event.cancel_tool = f"blocked by cost guard: human approval for '{device}' was not granted"


class QubitLimit:
    """Blocks run_qaoa before the simulator is called when the problem needs more qubits than allowed."""

    def __init__(self, store: LabStore, max_qubits: int):
        if max_qubits < 1:
            raise ValueError("max_qubits must be >= 1")
        self.store = store
        self.max_qubits = max_qubits

    def required_qubits(self, problem_id: str) -> int:
        return build_qubo(reduce(self.store.problem(problem_id).problem)).n_vars  # layers do not change the qubit count

    def check(self, event: BeforeToolCallEvent) -> None:
        if event.cancel_tool or event.tool_use["name"] not in EXECUTING_TOOLS:
            return
        problem_id = (event.tool_use.get("input") or {}).get("problem_id")
        try:
            qubits = self.required_qubits(problem_id)
        except UnknownId:
            return  # the tool itself reports unknown_id; nothing would run
        if qubits > self.max_qubits:
            event.cancel_tool = (
                f"blocked by qubit limit: problem '{problem_id}' needs {qubits} qubits, limit is {self.max_qubits}. "
                "Use solve_classical or reduce the problem."
            )


# --- phase C: provenance + notebook (AfterToolCallEvent) -----------------------------------------------------------

NOTEBOOK_STATE_KEY = "quantum_lab_notebook"


def _payload(result: dict) -> dict | None:
    """The tool's JSON result: Strands' @tool puts a dict result in a text block as JSON."""
    for c in (result or {}).get("content", []):
        if "text" in c:
            try:
                value = json.loads(c["text"])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return None


def _outcome(event: AfterToolCallEvent, payload: dict | None) -> str:
    if event.cancel_message:
        return "blocked"
    if event.exception is not None or (event.result or {}).get("status") == "error":
        return "failed"
    if isinstance(payload, dict) and payload.get("ok") is False:
        return "failed"
    return "completed"


class ProvenanceHook:
    """Attaches provenance (computed from the plugin's own store, not from the LLM) to every lab tool result."""

    def __init__(self, store: LabStore, tool_names: Iterable[str]):
        self.store = store
        self.tool_names = frozenset(tool_names)

    def label(self, event: AfterToolCallEvent) -> None:
        tool = event.tool_use["name"]
        if tool not in self.tool_names:
            return
        result = event.result or {}
        payload = _payload(result)
        prov = provenance.compute(tool, payload, self.store, _outcome(event, payload))
        content = list(result.get("content", []))
        if payload is not None:
            labelled = {**payload, "provenance": prov}
            idx = next(i for i, c in enumerate(content) if "text" in c and _payload({"content": [c]}) is not None)
            content[idx] = {"text": json.dumps(labelled, ensure_ascii=False)}
        else:  # blocked/failed with a plain-text message: keep it and add the provenance as a JSON block
            content.append({"json": {"provenance": prov}})
        event.result = {**result, "content": content}


def provenance_of(result: dict) -> dict | None:
    payload = _payload(result)
    if isinstance(payload, dict) and isinstance(payload.get("provenance"), dict):
        return payload["provenance"]
    for c in (result or {}).get("content", []):
        if isinstance(c.get("json"), dict) and isinstance(c["json"].get("provenance"), dict):
            return c["json"]["provenance"]
    return None


class NotebookHook:
    """Records one structured entry per lab tool call, after provenance, and mirrors the notebook into agent.state."""

    def __init__(self, notebook: Notebook, tool_names: Iterable[str]):
        self.notebook = notebook
        self.tool_names = frozenset(tool_names)

    def record(self, event: AfterToolCallEvent) -> None:
        tool = event.tool_use["name"]
        if tool not in self.tool_names:
            return
        result = event.result or {}
        payload = _payload(result)
        status = _outcome(event, payload)
        prov = provenance_of(result) or {"label": "none", "assigned_by": None, "evidence": "provenance hook did not label this result"}
        reason = event.cancel_message or (str(event.exception) if event.exception is not None else None)
        if reason is None and status == "failed":
            reason = (payload or {}).get("message") or next((c["text"] for c in result.get("content", []) if "text" in c), None)
        self.notebook.record(
            tool_use_id=event.tool_use["toolUseId"], tool=tool, status=status, params=event.tool_use.get("input") or {},
            payload=payload, provenance=prov, duration_s=event.duration, reason=reason,
        )
        event.agent.state.set(NOTEBOOK_STATE_KEY, self.notebook.to_dict())
