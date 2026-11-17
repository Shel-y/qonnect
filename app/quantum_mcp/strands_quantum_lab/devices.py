# SPDX-License-Identifier: GPL-3.0-only
"""Device classification (no Strands dependency): used by lab.py and by the safety hooks."""

from __future__ import annotations

LOCAL_DEVICES = frozenset({"local"})
MANAGED_SIMULATORS = frozenset({"sv1", "dm1", "tn1"})


def normalize_device(device: object) -> str:
    return str(device if device is not None else "local").strip().lower()


def classify_device(device: object) -> str:
    """'local' (free), 'paid' (Braket managed simulator or QPU) or 'unknown' (treated as blocked)."""
    d = normalize_device(device)
    if d in LOCAL_DEVICES:
        return "local"
    if d in MANAGED_SIMULATORS or d.startswith("qpu:") or ("arn:aws:braket" in d and ("/qpu/" in d or "/quantum-simulator/" in d)):
        return "paid"
    return "unknown"
