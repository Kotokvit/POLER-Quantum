"""The quantum-sampled POLER engine.

Same cognitive cycle as :class:`poler_quantum.core.PolerEngine`, but the
update acquires an *exploration term sampled from the Born distribution*
of the POLER ansatz:

    dp = Pi_Lambda (-grad_F + gamma_res * grad_res)          # classical part
         + tau * S(p)                                        # free dynamics
         + sigma_q * (mean of q_shots Born samples)          # quantum part

The quantum proposal ``s in {-1,+1}^dim`` is drawn by preparing the ansatz
from the *current* state and measuring it. Because of the resonance
entanglement layer, the coordinates of ``s`` are correlated -- the agent
explores in a structured, history-shaped way rather than with white noise.

The quantum engine therefore demonstrates the core claim of
POLER-Quantum: **cognition as measurement**. Attention (the free-energy
gradient) proposes; the quantum resonance disposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_aer import AerSimulator

from ..core.engine import PolerConfig, PolerEngine, StepResult
from .ansatz import PolerAnsatz


@dataclass
class QuantumConfig(PolerConfig):
    """POLER engine configuration + quantum parameters."""

    mode: str = "A"              # ansatz mode: A | B | C
    entanglement: str = "cx"     # resonance gate: cx | cz
    sigma_q: float = 0.05        # strength of the quantum exploration term
    q_shots: int = 8             # Born samples averaged per step
    q_seed: int | None = 42      # simulator seed (reproducibility)


@dataclass
class QuantumStepResult(StepResult):
    """Step diagnostics + quantum-specific fields."""

    quantum_proposal: np.ndarray | None = None   # mean Born sample
    born_entropy: float = float("nan")           # attention entropy


class QuantumPolerEngine(PolerEngine):
    """POLER[n] engine with a quantum-sampled exploration term."""

    def __init__(self, config: QuantumConfig | None = None,
                 Jc: np.ndarray | None = None,
                 G: np.ndarray | None = None,
                 J: np.ndarray | None = None,
                 D: np.ndarray | None = None) -> None:
        if config is None:
            config = QuantumConfig()
        super().__init__(config=config, Jc=Jc, G=G, J=J, D=D)
        self.qcfg: QuantumConfig = config
        self.ansatz = PolerAnsatz(self.cfg.dim, mode=self.qcfg.mode,
                                  entanglement=self.qcfg.entanglement)
        self.simulator = AerSimulator()
        self._shot_counter = 0

    # -- quantum proposal ----------------------------------------------------

    def quantum_proposal(self) -> np.ndarray:
        """Sample ``q_shots`` Born proposals and average them."""
        seed = None
        if self.qcfg.q_seed is not None:
            # deterministic per-step seed
            seed = self.qcfg.q_seed + self._shot_counter
            self._shot_counter += 1
        samples = self.ansatz.sample(
            self.simulator, self.p,
            gamma=self.cfg.gamma_res, kappa=self.cfg.kappa,
            shots=self.qcfg.q_shots, seed=seed)
        return samples.mean(axis=0)

    def attention_entropy(self) -> float:
        """Born entropy of the current ansatz (diagnostic)."""
        probs = self.ansatz.born_probabilities(
            self.p, gamma=self.cfg.gamma_res, kappa=self.cfg.kappa)
        return PolerAnsatz.born_entropy(probs)

    # -- the cognitive step ------------------------------------------------------

    def step(self, o_t: np.ndarray) -> QuantumStepResult:
        self.ensure_ready()
        omega = self.perception.omega(o_t)

        # significance (epsilon) -- attention modulation
        d_omega = (omega - self._last_omega) if self._last_omega is not None \
            else np.zeros_like(omega)
        eps, eps_hat = self.epsilon.update(d_omega)
        eta_eff = self.cfg.eta * float(np.exp(self.cfg.beta * np.tanh(eps_hat)))

        # classical cognition
        f_before = self.free_energy.value(self.p, omega)
        grad_F = self.free_energy.grad(self.p, omega)
        grad_res = self.resonance.gradient(self.p)
        sign = 1.0 if self.cfg.resonance_mode == "novelty" else -1.0

        # quantum proposal (Born sampling of the current state)
        proposal = self.quantum_proposal()
        entropy = self.attention_entropy()

        Pi = self.projector.matrix()
        raw = Pi @ (-grad_F + sign * self.cfg.gamma_res * grad_res)
        drift = self.cfg.tau * self.dynamics.drift(self.p, Pi)
        quantum = self.qcfg.sigma_q * (Pi @ proposal)
        dp = raw + drift + quantum

        p_next = np.clip(self.p + eta_eff * dp,
                         -self.cfg.clip, self.cfg.clip)
        f_after = self.free_energy.value(p_next, omega)

        self.resonance.push(omega)
        self._last_omega = omega
        self.p = p_next
        self.t += 1

        return QuantumStepResult(
            t=self.t,
            p=p_next.copy(),
            omega=omega.copy(),
            eps=eps,
            eps_hat=float(eps_hat),
            eta_eff=float(eta_eff),
            free_energy=f_before,
            free_energy_next=f_after,
            grad_norm=float(np.linalg.norm(raw)),
            quantum_proposal=proposal.copy(),
            born_entropy=entropy,
        )
