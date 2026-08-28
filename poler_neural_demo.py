"""
Small demonstration that turns the POLER analytic formula into a neural
loop with simultaneous observation / thinking / reflection streams.

The script feeds a couple of stimuli into the network and then lets it
run without input so that self-generated thoughts continue to emerge.
"""
from __future__ import annotations

import numpy as np

from poler_neural_net import PolerNeuralCore, format_state


if __name__ == "__main__":
    core = PolerNeuralCore(input_dim=6, latent_dim=24, thought_dim=12, kappa=1.1, rho=0.92, alpha=0.6, seed=42)

    # Synthetic stimuli encode three channels: pattern, observation, reflection
    stimuli = [
        np.array([0.8, 0.1, -0.2, 0.3, -0.4, 0.0]),
        np.array([-0.5, 0.4, 0.7, -0.1, 0.2, -0.3]),
    ]

    history = core.run_sequence(stimuli, steps_without_input=4)

    for step, state in enumerate(history):
        print(format_state(step, state))
        print("-" * 80)
