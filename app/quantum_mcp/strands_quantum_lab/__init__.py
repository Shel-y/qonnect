# SPDX-License-Identifier: GPL-3.0-only
"""strands-quantum-lab: hybrid classical-quantum optimization tools for Strands Agents.

Public API (everything a developer needs; other modules are internal):

    from strands_quantum_lab import (
        QuantumLabPlugin, TOOL_NAMES,      # the Strands plugin and its six tool names
        APPROVE, COST_INTERRUPT,           # answer the cost guard's human-approval interrupt
        Notebook, NOTEBOOK_STATE_KEY,      # read / export the experiment notebook
        LabStore,                          # optional: inject your own problem/run registry
        SelectionProblem, Item,            # the supported problem model
    )
    import strands_quantum_lab.lab         # the six operations as plain functions, without Strands

Names that depend on Strands are imported lazily, so ``strands_quantum_lab.lab`` stays Strands-free.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from strands_quantum_lab.notebook import Notebook
from strands_quantum_lab.problem import Item, SelectionProblem
from strands_quantum_lab.store import LabStore

try:
    __version__ = version("strands-quantum-lab")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without installing
    __version__ = "0.0.0+unknown"

_LAZY = {
    "QuantumLabPlugin": "strands_quantum_lab.plugin",
    "TOOL_NAMES": "strands_quantum_lab.plugin",
    "APPROVE": "strands_quantum_lab.hooks",
    "COST_INTERRUPT": "strands_quantum_lab.hooks",
    "NOTEBOOK_STATE_KEY": "strands_quantum_lab.hooks",
}

__all__ = [
    "APPROVE",
    "COST_INTERRUPT",
    "NOTEBOOK_STATE_KEY",
    "TOOL_NAMES",
    "Item",
    "LabStore",
    "Notebook",
    "QuantumLabPlugin",
    "SelectionProblem",
    "__version__",
]


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'strands_quantum_lab' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
