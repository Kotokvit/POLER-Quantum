"""Quantum side of POLER-Quantum.

The POLER[n] state is encoded into a parametrised quantum circuit
("POLER ansatz") and the cognitive update is *sampled* from its Born
distribution on a Qiskit Aer simulator.
"""

from .ansatz import PolerAnsatz  # noqa: F401
from .engine import QuantumPolerEngine, QuantumConfig  # noqa: F401

__all__ = ["PolerAnsatz", "QuantumPolerEngine", "QuantumConfig"]
