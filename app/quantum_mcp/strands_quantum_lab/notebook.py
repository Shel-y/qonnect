# SPDX-License-Identifier: GPL-3.0-only
"""Experiment notebook: one structured entry per lab tool call. No Strands import.

Schema (version 1), stable and JSON-serializable:

    {
      "schema": "strands-quantum-lab/notebook",
      "schema_version": 1,
      "entries": [
        {
          "entry": 1,                     # 1-based sequence within this notebook
          "tool_use_id": "tooluse_...",   # Strands tool-use id (unique per call)
          "tool": "run_qaoa",
          "status": "completed" | "failed" | "blocked",
          "reason": str | null,           # block message or error message
          "error_code": str | null,       # lab error code when the operation returned ok=false
          "params": {...},                # tool input; bulky fields reduced (see _params)
          "problem_id": str | null,
          "run_id": str | null,
          "parent_run_id": str | null,
          "summary": {...},               # small per-tool summary, never the full result
          "duration_s": float | null,     # Strands-measured execution time (null when blocked)
          "provenance": {...},            # as computed by the provenance hook (see provenance.py)
          "recorded_at": "ISO-8601 UTC"
        }
      ]
    }
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

SCHEMA = "strands-quantum-lab/notebook"
SCHEMA_VERSION = 1
ENTRY_KEYS = (
    "entry", "tool_use_id", "tool", "status", "reason", "error_code", "params", "problem_id", "run_id",
    "parent_run_id", "summary", "duration_s", "provenance", "recorded_at",
)
STATUSES = ("completed", "failed", "blocked")
MAX_DESCRIPTION = 200


def _params(tool: str, params: dict) -> dict:
    p = dict(params or {})
    if tool == "formulate_problem":
        items = p.get("items")
        if isinstance(items, list):
            p["items"] = {"count": len(items), "ids": [i.get("id") for i in items if isinstance(i, dict)]}
        if isinstance(p.get("conflicts"), list):
            p["conflicts"] = {"count": len(p["conflicts"])}
        if isinstance(p.get("description"), str) and len(p["description"]) > MAX_DESCRIPTION:
            p["description"] = p["description"][:MAX_DESCRIPTION] + "…"
    return p


def _pick(d: dict | None, *keys: str) -> dict:
    return {k: d.get(k) for k in keys if isinstance(d, dict) and k in d}


def summarize(tool: str, payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    if not payload.get("ok"):
        return _pick(payload, "error", "message")
    if tool == "formulate_problem":
        prob = payload.get("problem", {})
        return {
            "items": len(prob.get("items", [])), "budget": prob.get("budget"), "conflicts": len(prob.get("conflicts", [])),
            **_pick(payload.get("qubo"), "qaoa_qubits", "within_local_simulation_limit"),
        }
    if tool == "solve_classical":
        best = (payload.get("solutions") or [None])[0]
        return {"method": payload.get("method"), "optimal": payload.get("optimal"), "solutions": len(payload.get("solutions", [])),
                "best": _pick(best, "selected", "objective", "cost")}
    if tool == "run_qaoa":
        return {**_pick(payload, "device", "qubits", "layers", "shots", "circuit_evaluations", "expected_energy", "feasible_shot_fraction"),
                "best_feasible_sample": _pick(payload.get("best_feasible_sample"), "selected", "objective"),
                "timings_ms": payload.get("timings_ms")}
    if tool == "hybrid_shortlist":
        return {"chosen": _pick(payload.get("chosen"), "selected", "objective", "origin"), "shortlist_size": len(payload.get("shortlist", [])),
                **_pick(payload, "entries_needing_classical_help")}
    if tool == "analyze_results":
        c = payload.get("contributions", {})
        return {
            "gain_over_qaoa_best": (c.get("classical_refinement") or {}).get("objective_gain_over_qaoa_best"),
            "constraints_satisfied": (c.get("combined") or {}).get("constraints_satisfied"),
            **_pick(payload.get("verification"), "result_is_optimal", "optimality_gap"),
        }
    if tool == "estimate_resources":
        return {**_pick(payload, "device", "qubits", "gates", "depth", "executed_anything"),
                "estimated_cost_usd": (payload.get("on_hardware") or {}).get("estimated_cost_usd")}
    return {}


class Notebook:
    def __init__(self, entries: list[dict] | None = None):
        self.entries: list[dict] = list(entries or [])

    def record(
        self, *, tool_use_id: str, tool: str, status: str, params: dict, payload: dict | None, provenance: dict,
        duration_s: float | None, reason: str | None,
    ) -> dict:
        if status not in STATUSES:
            raise ValueError(f"invalid status {status}")
        if any(e["tool_use_id"] == tool_use_id for e in self.entries):
            raise ValueError(f"tool call {tool_use_id} is already recorded")
        ok_payload = payload if isinstance(payload, dict) else {}
        entry = {
            "entry": len(self.entries) + 1,
            "tool_use_id": tool_use_id,
            "tool": tool,
            "status": status,
            "reason": reason,
            "error_code": None if ok_payload.get("ok", True) else ok_payload.get("error"),
            "params": _params(tool, params),
            "problem_id": ok_payload.get("problem_id") or (params or {}).get("problem_id"),
            "run_id": ok_payload.get("run_id") if ok_payload.get("ok") else None,
            "parent_run_id": ok_payload.get("qaoa_run_id") if tool == "hybrid_shortlist" else None,
            "summary": summarize(tool, payload),
            "duration_s": None if duration_s is None else round(duration_s, 6),
            "provenance": provenance,
            "recorded_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
        json.dumps(entry)  # fail loudly if anything is not serializable
        self.entries.append(entry)
        return entry

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "entries": [dict(e) for e in self.entries]}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Notebook:
        data = json.loads(text)
        if data.get("schema") != SCHEMA or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("not a strands-quantum-lab notebook (schema version 1)")
        for e in data.get("entries", []):
            missing = [k for k in ENTRY_KEYS if k not in e]
            if missing or e["status"] not in STATUSES:
                raise ValueError(f"invalid notebook entry {e.get('entry')}: missing {missing}")
        return cls(data["entries"])
