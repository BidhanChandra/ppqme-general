"""Partial-polaron weighting function and transformation parameters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray

TransformKind = Literal['partial', 'none']

@dataclass(frozen=True)
class PartialPolaronTransform:
    omega_h: float = 0.31
    alpha: float = 2.0
    kind: TransformKind = 'partial'
    frequency_points: int = 20000
    omega_max: float = 0.4
    time_chunk: int = 128

    def validate(self) -> None:
        if self.kind not in {'partial','none'}:
            raise ValueError("kind must be 'partial' or 'none'")
        if self.kind == 'partial' and self.omega_h <= 0:
            raise ValueError('omega_h must be positive')
        if self.alpha <= 0 or self.frequency_points < 100:
            raise ValueError('invalid alpha/frequency_points')
        if self.omega_max <= 0 or self.time_chunk < 1:
            raise ValueError('invalid omega_max/time_chunk')


def weighting_function(omega: ArrayLike, transform: PartialPolaronTransform) -> NDArray[np.float64]:
    transform.validate()
    w = np.asarray(omega, dtype=float)
    if transform.kind == 'none':
        return np.zeros_like(w)
    x = np.maximum(w/transform.omega_h, 0.0)
    return -np.expm1(-(x**transform.alpha))
