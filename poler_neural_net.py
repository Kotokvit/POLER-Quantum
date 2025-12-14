"""
Neural implementation of the POLER[n] cognitive loop with parallel
observation, thinking, and reflection streams.

The model maintains a latent state that is updated even when no
external stimulus is provided, allowing "thoughts" to emerge from
previous context and resonance feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Utility activation functions -------------------------------------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PolerState:
    """Container for the dynamic state of the POLER core."""

    latent: np.ndarray
    thought: np.ndarray
    observation: np.ndarray
    reflection: np.ndarray
    resonance: np.ndarray
    energy: float


class PolerNeuralCore:
    """Neural analogue of the POLER cycle.

    The architecture exposes three parallel heads (observation, thought,
    reflection) that are computed in a single forward call, emulating the
    user's diagram with simultaneous flows and real-time feedback.
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 32,
        thought_dim: int = 16,
        kappa: float = 0.8,
        rho: float = 0.9,
        alpha: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Encoder from stimulus + recurrent context to latent
        self.W_enc = rng.standard_normal((input_dim + thought_dim * 2, latent_dim)) * 0.25
        self.b_enc = np.zeros(latent_dim)

        # Heads: simultaneous computation of observation, thought, reflection
        self.W_obs = rng.standard_normal((latent_dim, thought_dim)) * 0.2
        self.b_obs = np.zeros(thought_dim)

        self.W_thought = rng.standard_normal((latent_dim, thought_dim)) * 0.2
        self.b_thought = np.zeros(thought_dim)

        self.W_refl = rng.standard_normal((latent_dim + thought_dim * 2, thought_dim)) * 0.2
        self.b_refl = np.zeros(thought_dim)

        # Feedback gates to modulate the next latent state
        self.W_feedback = rng.standard_normal((thought_dim * 3, latent_dim)) * 0.1
        self.b_feedback = np.zeros(latent_dim)

        # Resonance/energy hyper-parameters
        self.kappa = kappa
        self.rho = rho
        self.alpha = alpha

        # Initial state
        zero_thought = np.zeros(thought_dim)
        zero_latent = np.zeros(latent_dim)
        self.state = PolerState(
            latent=zero_latent,
            thought=zero_thought,
            observation=zero_thought,
            reflection=zero_thought,
            resonance=np.zeros(thought_dim),
            energy=0.0,
        )

    def _encode(self, stimulus: np.ndarray, prev_thought: np.ndarray, prev_reflection: np.ndarray) -> np.ndarray:
        combined = np.concatenate([stimulus, prev_thought, prev_reflection])
        return _tanh(combined @ self.W_enc + self.b_enc)

    def _heads(self, latent: np.ndarray):
        observation = _tanh(latent @ self.W_obs + self.b_obs)
        thought = _tanh(latent @ self.W_thought + self.b_thought)
        reflection_in = np.concatenate([latent, observation, thought])
        reflection = _tanh(reflection_in @ self.W_refl + self.b_refl)
        return observation, thought, reflection

    def _feedback(self, observation: np.ndarray, thought: np.ndarray, reflection: np.ndarray) -> np.ndarray:
        stacked = np.concatenate([observation, thought, reflection])
        return _sigmoid(stacked @ self.W_feedback + self.b_feedback)

    def step(self, stimulus: np.ndarray | None = None, noise_scale: float = 0.05) -> PolerState:
        """Perform one POLER[n] cycle.

        Args:
            stimulus: External input vector. If None, an internal noise vector
                is used so that the system can continue to generate thoughts
                without new sensory data ("мысль возникала сама собой").
            noise_scale: Standard deviation of the internal noise when
                stimulus is absent.
        """

        prev = self.state
        if stimulus is None:
            stimulus = np.random.standard_normal(len(self.W_enc))[: self.W_enc.shape[0] - prev.thought.size * 2]
            stimulus *= noise_scale

        latent = self._encode(stimulus, prev.thought, prev.reflection)
        observation, thought, reflection = self._heads(latent)

        # Resonance captures decayed memory of prior thoughts
        resonance = self.rho * prev.resonance + self.alpha * thought

        # Energy measures divergence between observation and thought
        energy = float(self.kappa * np.linalg.norm(observation - thought) ** 2)

        # Feedback modulates latent dynamics in real time
        feedback_gate = self._feedback(observation, thought, reflection)
        latent = latent * feedback_gate + prev.latent * (1.0 - feedback_gate)

        self.state = PolerState(
            latent=latent,
            thought=thought,
            observation=observation,
            reflection=reflection,
            resonance=resonance,
            energy=energy,
        )
        return self.state

    def run_sequence(self, stimuli: list[np.ndarray] | None, steps_without_input: int = 3) -> list[PolerState]:
        """Run a sequence of stimuli and then continue thinking on its own."""

        history: list[PolerState] = []
        if stimuli:
            for stim in stimuli:
                history.append(self.step(stimulus=stim))

        for _ in range(steps_without_input):
            history.append(self.step(stimulus=None))
        return history


def format_state(step: int, state: PolerState) -> str:
    """Nicely format the state for logging/demo purposes."""

    obs = np.array2string(state.observation, precision=3, floatmode="fixed")
    thought = np.array2string(state.thought, precision=3, floatmode="fixed")
    refl = np.array2string(state.reflection, precision=3, floatmode="fixed")
    res = np.array2string(state.resonance, precision=3, floatmode="fixed")
    return (
        f"t={step:02d} | ε={state.energy:0.4f}\n"
        f"  observation: {obs}\n"
        f"  thought:      {thought}\n"
        f"  reflection:   {refl}\n"
        f"  resonance:    {res}"
    )
