# SPDX-License-Identifier: GPL-3.0-only
"""Strands Agents adapter: exposes the six lab operations (``lab.py``) as tools. No optimization logic lives here.

Scope: selection problems (maximize total value within an integer budget, avoiding conflicting pairs, with optional
mandatory items). QAOA runs only on Amazon Braket LocalSimulator.
Hooks: CostGuard and QubitLimit (BeforeToolCallEvent); Provenance then Notebook (AfterToolCallEvent, explicit priorities).
"""

from __future__ import annotations

from collections.abc import Iterable

from strands import tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookOrder
from strands.plugins import Plugin, hook

from strands_quantum_lab import lab
from strands_quantum_lab.hooks import NOTEBOOK_STATE_KEY, CostGuard, NotebookHook, ProvenanceHook, QubitLimit
from strands_quantum_lab.notebook import Notebook
from strands_quantum_lab.qaoa import MAX_SIM_QUBITS
from strands_quantum_lab.store import LabStore

TOOL_NAMES = ("formulate_problem", "solve_classical", "run_qaoa", "hybrid_shortlist", "analyze_results", "estimate_resources")
# AfterToolCallEvent runs callbacks by priority (lower first) and only reverses registration order *within* a priority;
# distinct explicit priorities make Provenance run before Notebook regardless of declaration/registration order.
PROVENANCE_ORDER = HookOrder.DEFAULT - 20
NOTEBOOK_ORDER = HookOrder.DEFAULT - 10


class QuantumLabPlugin(Plugin):
    """Hybrid classical-quantum optimization tools for any Strands agent. Use one instance per agent."""

    name = "strands-quantum-lab"

    def __init__(self, store: LabStore | None = None, *, max_qubits: int = 16, allowed_paid_devices: Iterable[str] = ()):
        """
        Args:
            store: Registry for problems and runs (a new one per instance by default).
            max_qubits: QubitLimit: run_qaoa is blocked above this many qubits (1 to the local simulation limit).
            allowed_paid_devices: CostGuard: paid devices (e.g. "sv1") that may be *requested*, always behind human
                approval. Empty by default: every paid/QPU device is blocked. Approval never executes anything in this
                version (paid devices return not_implemented).
        """
        if not 1 <= max_qubits <= MAX_SIM_QUBITS:
            raise ValueError(f"max_qubits must be between 1 and {MAX_SIM_QUBITS}")
        self.store = store or LabStore()
        self.qubit_limit = QubitLimit(self.store, max_qubits)
        self.cost_guard = CostGuard(allowed_paid_devices)
        self.notebook = Notebook()
        self.provenance_hook = ProvenanceHook(self.store, TOOL_NAMES)
        self.notebook_hook = NotebookHook(self.notebook, TOOL_NAMES)
        self._agent = None
        super().__init__()

    def init_agent(self, agent) -> None:
        """Registers the AfterToolCallEvent hooks with explicit priorities (the @hook decorator has no priority)."""
        if self._agent is not None and self._agent is not agent:
            raise ValueError("a QuantumLabPlugin instance belongs to one agent; create one plugin per agent")
        self._agent = agent
        agent.add_hook(self.provenance_hook.label, AfterToolCallEvent, order=PROVENANCE_ORDER)
        agent.add_hook(self.notebook_hook.record, AfterToolCallEvent, order=NOTEBOOK_ORDER)
        agent.state.set(NOTEBOOK_STATE_KEY, self.notebook.to_dict())

    def export_notebook(self) -> str:
        """The notebook as JSON (schema in notebook.py). Read it back with Notebook.from_json."""
        return self.notebook.to_json()

    # Hooks run in declaration order: the qubit limit first, so no approval is requested for a call that would be blocked anyway.
    @hook
    def enforce_qubit_limit(self, event: BeforeToolCallEvent) -> None:
        self.qubit_limit.check(event)

    @hook
    def enforce_cost_guard(self, event: BeforeToolCallEvent) -> None:
        self.cost_guard.check(event)

    @tool
    def formulate_problem(
        self,
        items: list[dict],
        budget: int,
        conflicts: list[list[str]] | None = None,
        name: str = "selection",
        description: str = "",
        problem_type: str = "selection",
    ) -> dict:
        """Validate a selection problem, build its QUBO and register it. Returns a problem_id for the other tools.

        Turn the user's description into this structure yourself; this tool does not read natural language.
        Only problem_type="selection" is supported: choose items maximizing total value with total cost <= budget,
        never selecting both items of a conflicting pair, and always selecting mandatory items.

        Args:
            items: List of {"id": str, "value": number >= 0, "cost": integer >= 0, "mandatory": bool (optional),
                "label": str (optional)}. At most 24 items.
            budget: Integer maximum total cost, in the same unit as the item costs.
            conflicts: Pairs of item ids that cannot both be selected, e.g. [["a", "b"]].
            name: Short name for the problem.
            description: The original natural-language description, kept for the record.
            problem_type: Must be "selection".
        """
        return lab.formulate_problem(self.store, items, budget, conflicts, name, description, problem_type)

    @tool
    def solve_classical(self, problem_id: str, method: str = "greedy", top_k: int = 1) -> dict:
        """Classical reference solution for a registered problem.

        Args:
            problem_id: Id returned by formulate_problem.
            method: "greedy" (default, fast heuristic) or "exact" (exhaustive, optimal —
                only use when the user explicitly asks for the optimal or exact solution).
            top_k: With method="exact", how many best solutions to return (1-20).
        """
        return lab.solve_classical(self.store, problem_id, method, top_k)

    @tool
    def run_qaoa(self, problem_id: str, layers: int = 1, shots: int = 1000, maxiter: int = 25, seed: int = 7, device: str = "local") -> dict:
        """Run QAOA (with its classical variational loop) on Amazon Braket LocalSimulator. Returns a run_id.

        LocalSimulator is a classical simulation of a quantum circuit, not quantum hardware.

        Args:
            problem_id: Id returned by formulate_problem.
            layers: QAOA depth p (1-3).
            shots: Measurement shots for the final circuit (1-100000).
            maxiter: Maximum iterations of the classical optimizer (1-500). Default is 25
                (fast exploration). Use higher values only when the user asks for a more
                thorough optimization.
            seed: Seed for the initial angles.
            device: "local" (Braket LocalSimulator). Paid devices (sv1, dm1, tn1, QPUs) are blocked by the cost guard
                and are not implemented in this version.
        """
        return lab.run_qaoa(self.store, problem_id, layers, shots, maxiter, seed, device)

    @tool
    def hybrid_shortlist(self, qaoa_run_id: str, k: int = 5, polish: bool = True) -> dict:
        """Hybrid step: take QAOA's measured candidates, let classical repair/polish them, keep the top-k and choose one.

        Args:
            qaoa_run_id: run_id returned by run_qaoa.
            k: Shortlist size (1-20).
            polish: Apply classical local search to the candidates.
        """
        return lab.hybrid_shortlist(self.store, qaoa_run_id, k, polish)

    @tool
    def analyze_results(self, problem_id: str, qaoa_run_id: str = "", shortlist_run_id: str = "") -> dict:
        """Explain results, separating QAOA's contribution, the classical refinement and the combined result.

        The exact classical optimum is included only as a verification reference.

        Args:
            problem_id: Id returned by formulate_problem.
            qaoa_run_id: run_id from run_qaoa (optional if shortlist_run_id is given).
            shortlist_run_id: run_id from hybrid_shortlist (optional).
        """
        return lab.analyze_results(self.store, problem_id, qaoa_run_id or None, shortlist_run_id or None)

    @tool
    def estimate_resources(
        self,
        problem_id: str,
        layers: int = 1,
        shots: int = 1000,
        circuit_evaluations: int = 25,
        device: str = "local",
        price_per_task_usd: float | None = None,
        price_per_shot_usd: float | None = None,
    ) -> dict:
        """Estimate qubits, gates, circuit depth and potential QPU cost for a registered problem. Executes nothing.

        Cost is computed only if current prices for the target device are given; none are assumed.

        Args:
            problem_id: Id returned by formulate_problem.
            layers: QAOA depth p (1-3).
            shots: Shots per circuit evaluation.
            circuit_evaluations: Expected evaluations of the variational loop. Default is 25,
                matching the default maxiter of run_qaoa.
            device: Target device label, recorded as given.
            price_per_task_usd: Current per-task price of the target device, if known.
            price_per_shot_usd: Current per-shot price of the target device, if known.
        """
        return lab.estimate_resources(self.store, problem_id, layers, shots, circuit_evaluations, device, price_per_task_usd, price_per_shot_usd)
