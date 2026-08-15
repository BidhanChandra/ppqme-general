"""Common spectral-density constructors.

The core solver accepts any callable J(omega).  These helpers cover common
harmonic-bath choices without coupling the solver to a particular model.
All energies/frequencies are in eV.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DebyeSpectralDensity:
    reorganization_energy: float
    cutoff: float

    def __post_init__(self):
        if self.reorganization_energy < 0:
            raise ValueError("reorganization_energy must be >= 0")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0")

    def __call__(self, omega):
        w = np.asarray(omega, dtype=float)
        lam, wc = self.reorganization_energy, self.cutoff
        return 2.0 * lam * w * wc / (w*w + wc*wc)


@dataclass(frozen=True)
class OhmicExponentialSpectralDensity:
    strength: float
    cutoff: float

    def __post_init__(self):
        if self.strength < 0:
            raise ValueError("strength must be >= 0")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0")

    def __call__(self, omega):
        w = np.asarray(omega, dtype=float)
        return self.strength * w * np.exp(-w / self.cutoff)


def from_dict(spec: dict):
    """Build a spectral-density callable from a JSON-compatible dictionary."""
    kind = str(spec.get("type", "debye")).lower()
    if kind in {"debye", "drude", "lorentz-drude"}:
        return DebyeSpectralDensity(
            reorganization_energy=float(spec["reorganization_energy"]),
            cutoff=float(spec["cutoff"]),
        )
    if kind in {"ohmic", "ohmic-exponential"}:
        return OhmicExponentialSpectralDensity(
            strength=float(spec["strength"]),
            cutoff=float(spec["cutoff"]),
        )
    raise ValueError(f"unknown spectral density type: {kind!r}")
