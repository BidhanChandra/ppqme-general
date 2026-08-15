"""Independent harmonic-bath functions entering Jang's p-PQME.

Implements the site-resolved forms of Jang 2022 Eqs. (42)-(44) and (15).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .transform import PartialPolaronTransform, weighting_function

SpectralDensity = Callable[[NDArray[np.float64]], NDArray[np.float64]]

@dataclass(frozen=True)
class BathNumerics:
    beta: float
    rate_tmax: float
    rate_dt: float

    def validate(self):
        if self.beta <= 0:  # np.inf is valid for T=0
            raise ValueError('beta must be positive or infinity')
        if self.rate_tmax <= 0 or self.rate_dt <= 0 or self.rate_dt >= self.rate_tmax:
            raise ValueError('invalid time grid')


def _coth_stable(x: ArrayLike):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-5
    xs = x[small]
    out[small] = 1/xs + xs/3 - xs**3/45
    out[~small] = 1/np.tanh(x[~small])
    return out


def independent_bath_components_grid(times, spectral_densities: Sequence[SpectralDensity], beta: float, transform: PartialPolaronTransform):
    t = np.asarray(times, dtype=float)
    transform.validate()
    nw = transform.frequency_points
    dw = transform.omega_max/nw
    omega = (np.arange(nw, dtype=float)+0.5)*dw
    Wh = weighting_function(omega, transform)
    one = 1.0-Wh
    thermal = _coth_stable(0.5*beta*omega)

    n = len(spectral_densities)
    nt = t.size
    phi = np.empty((n,nt), complex)
    mixed = np.empty_like(phi)
    linear = np.empty_like(phi)
    lambda_partial = np.empty(n)

    for j, Jfun in enumerate(spectral_densities):
        J = np.maximum(np.asarray(Jfun(omega), float), 0.0)
        wr_phi = (dw/np.pi)*J/omega**2*Wh**2*thermal
        wi_phi = (dw/np.pi)*J/omega**2*Wh**2
        wr_m = (dw/np.pi)*J/omega*Wh*one
        wi_m = wr_m*thermal
        wr_c = (dw/np.pi)*J*one**2*thermal
        wi_c = (dw/np.pi)*J*one**2
        for begin in range(0, nt, transform.time_chunk):
            stop = min(begin+transform.time_chunk, nt)
            phase = omega[:,None]*t[None,begin:stop]
            cs, sn = np.cos(phase), np.sin(phase)
            phi[j,begin:stop] = wr_phi@cs - 1j*(wi_phi@sn)
            mixed[j,begin:stop] = wr_m@cs - 1j*(wi_m@sn)
            linear[j,begin:stop] = wr_c@cs - 1j*(wi_c@sn)
        lambda_partial[j] = (dw/np.pi)*np.sum(J/omega*Wh*(2-Wh))

    return dict(phi_site=phi, mixed_site=mixed, linear_site=linear,
                lambda_partial=lambda_partial, omega_nodes=omega,
                transformed_fraction=Wh)
