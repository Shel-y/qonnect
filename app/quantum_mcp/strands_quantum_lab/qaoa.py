# SPDX-License-Identifier: GPL-3.0-only
"""QAOA on Amazon Braket LocalSimulator (adapted from Qonnect, LA-10).

Hybrid loop: the circuit runs on the simulator; a classical optimizer (scipy COBYLA) tunes its angles.
During tuning, expectations use the simulator's exact probability vector (shots=0) so the loop is
deterministic; the final circuit is then *measured* with shots, like it would be on hardware.

Provenance: ``simulated``. LocalSimulator is a classical simulation of a quantum circuit: it is not a
QPU run and not evidence of quantum advantage.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from strands_quantum_lab.problem import SelectionProblem
from strands_quantum_lab.qubo import Qubo, Reduction, build_qubo, energy_vector, reduce, to_ising

MAX_SIM_QUBITS = 22


class QuantumUnavailable(RuntimeError):
    """The optional [braket] extra is not installed."""


def quantum_available() -> bool:
    try:
        import braket.devices  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Sample:
    x: list[int]  # full assignment (fixed items included)
    count: int
    feasible: bool
    objective: float


@dataclass
class QaoaResult:
    n_qubits: int
    layers: int
    gammas: list[float]
    betas: list[float]
    expected_energy: float
    energy_history: list[float]
    circuit_evaluations: int
    shots: int
    samples: list[Sample]  # distinct measured item assignments, most frequent first
    timings_ms: dict[str, float]
    circuit: dict
    reduction: dict
    diagnostics: dict = field(default_factory=dict)
    provenance: str = "simulated"

    @property
    def best_feasible(self) -> Sample | None:
        feas = [s for s in self.samples if s.feasible]
        return max(feas, key=lambda s: (s.objective, s.count)) if feas else None


def build_circuit(n: int, h, jj, gammas, betas, scale: float, measure_probabilities: bool):
    from braket.circuits import Circuit

    circ = Circuit()
    for q in range(n):
        circ.h(q)
    for gamma, beta in zip(gammas, betas, strict=True):
        for (i, j), coupling in jj.items():
            circ.zz(i, j, 2 * gamma * coupling / scale)
        for q in range(n):
            if h[q]:
                circ.rz(q, 2 * gamma * h[q] / scale)
        for q in range(n):
            circ.rx(q, 2 * beta)
    if measure_probabilities:
        circ.probability()
    return circ


def run_qaoa(
    problem: SelectionProblem,
    layers: int = 1,
    shots: int = 1000,
    maxiter: int = 25,
    seed: int = 7,
    red: Reduction | None = None,
    qubo: Qubo | None = None,
) -> QaoaResult:
    if not quantum_available():
        raise QuantumUnavailable("install the [braket] extra: amazon-braket-sdk, numpy, scipy")
    import numpy as np
    from braket.devices import LocalSimulator
    from scipy.optimize import minimize

    t_form = time.perf_counter()
    red = red or reduce(problem)
    qubo = qubo or build_qubo(red)
    n = qubo.n_vars
    if n == 0:
        raise ValueError("nothing to optimize: the classical reduction fixed every item")
    if n > MAX_SIM_QUBITS:
        raise ValueError(f"{n} qubits exceeds the local simulation limit of {MAX_SIM_QUBITS}")
    h, jj, _ = to_ising(qubo)
    scale = max([abs(v) for v in h] + [abs(v) for v in jj.values()] + [1e-9])
    energies = energy_vector(qubo)
    formulation_ms = (time.perf_counter() - t_form) * 1000

    device = LocalSimulator("braket_sv")
    rng = np.random.default_rng(seed)
    history: list[float] = []
    sim_seconds = 0.0

    def expectation(theta) -> float:
        nonlocal sim_seconds
        t0 = time.perf_counter()
        circ = build_circuit(n, h, jj, theta[:layers], theta[layers:], scale, measure_probabilities=True)
        probs = np.asarray(device.run(circ, shots=0).result().values[0], dtype=np.float64)
        sim_seconds += time.perf_counter() - t0
        value = float(probs @ energies)
        history.append(value)
        return value

    t_opt = time.perf_counter()
    theta0 = rng.uniform(0.1, 1.0, size=2 * layers)
    res = minimize(expectation, theta0, method="COBYLA", options={"maxiter": maxiter})
    optimization_ms = (time.perf_counter() - t_opt) * 1000

    t_meas = time.perf_counter()
    final = build_circuit(n, h, jj, res.x[:layers], res.x[layers:], scale, measure_probabilities=False)
    counts = device.run(final, shots=shots).result().measurement_counts
    measurement_ms = (time.perf_counter() - t_meas) * 1000

    # Aggregate over slack qubits: what matters is the item assignment.
    by_items: Counter[tuple[int, ...]] = Counter()
    for bitstring, c in counts.items():
        by_items[tuple(int(b) for b in bitstring[: qubo.n_items])] += c
    samples = []
    for sub, c in by_items.most_common():
        x = red.lift(list(sub))
        samples.append(Sample(x=x, count=c, feasible=problem.is_feasible(x), objective=problem.objective(x)))

    return QaoaResult(
        n_qubits=n,
        layers=layers,
        gammas=[float(v) for v in res.x[:layers]],
        betas=[float(v) for v in res.x[layers:]],
        expected_energy=float(res.fun),
        energy_history=[round(v, 6) for v in history],
        circuit_evaluations=len(history),
        shots=shots,
        samples=samples,
        timings_ms={
            "formulation": round(formulation_ms, 3),
            "variational_loop": round(optimization_ms, 3),
            "of_which_simulation": round(sim_seconds * 1000, 3),
            "final_measurement": round(measurement_ms, 3),
        },
        circuit={
            "qubits": n,
            "item_qubits": qubo.n_items,
            "slack_qubits": qubo.n_slack,
            "two_qubit_terms": len(jj),
            "gates": len(final.instructions),
            "depth": int(final.depth),
        },
        reduction=red.summary(),
        diagnostics={"optimizer_success": bool(res.success), "optimizer_message": str(res.message)},
    )
