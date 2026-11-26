r"""GOA bath: secondary-mode discretisation, normal modes, lab-frame correlation function.

Secondary (Ohmic, exponential cutoff), N_s = N - 1 modes:
    w_j = -omega_c ln[(N_s + 1/2 - j)/N_s],   c_j = w_j sqrt(2 eta omega_c / (pi N_s))

Mass-weighted Hessian (counter-term on the primary coordinate q_0):
    K = [[Omega^2 + sum_j c_j^2/w_j^2, c^T], [c, diag(w_j^2)]] = U diag(w_q^2) U^T
Donor->acceptor shift  d = 2 K^{-1} (Omega^2 y0 e_0),  R_q = (U^T d)_q,
    E_r = 1/2 sum_q w_q^2 R_q^2 = 2 Omega^2 y0^2   (Schur complement).

Lab-frame bath correlation function of X = -sum_q w_q^2 R_q q_q:
    C(tau) = sum_q (w_q^3 R_q^2 / 2) [coth(beta w_q/2) cos(w_q tau) - i sin(w_q tau)],
    J(w)   = (pi/2) sum_q w_q^3 R_q^2 delta(w - w_q).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import ModelParams


def coth_stable(x):
    """coth(x) with a series for |x| < 1e-7."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-7
    xs = x[small]
    out[small] = 1.0 / xs + xs / 3.0 - xs ** 3 / 45.0
    out[~small] = 1.0 / np.tanh(x[~small])
    return out


def thermal_coth(beta: float, omega) -> np.ndarray:
    """coth(beta w / 2) = 2 n(w) + 1 ; equal to 1 at beta = inf."""
    omega = np.asarray(omega, dtype=float)
    if np.isinf(beta):
        return np.ones_like(omega)
    return coth_stable(0.5 * beta * omega)


def secondary_modes(N: int, eta: float, omega_c: float):
    """Log-uniform discretisation of J_sec(w) = eta w exp(-w/omega_c): returns (w_j, c_j), length N-1."""
    Ns = int(N) - 1
    j = np.arange(1, Ns + 1, dtype=float)
    w = -omega_c * np.log((Ns + 0.5 - j) / Ns)
    c = w * np.sqrt(2.0 * eta * omega_c / (np.pi * Ns))
    return w, c


@dataclass(frozen=True)
class BathModes:
    """Normal modes of the GOA bath."""
    omega: np.ndarray     # (N,) normal-mode frequencies w_q
    R: np.ndarray         # (N,) donor->acceptor displacement along each normal mode
    Er: float             # reorganisation energy 1/2 sum w^2 R^2

    @property
    def lab_weights(self) -> np.ndarray:
        """w_q^3 R_q^2 / 2 : weights of C(tau) (and of J(w) up to pi)."""
        return 0.5 * self.omega ** 3 * self.R ** 2


@lru_cache(maxsize=64)
def _normal_modes(Omega: float, y0: float, N: int, eta: float, omega_c: float) -> BathModes:
    wsec, csec = secondary_modes(N, eta, omega_c)
    K = np.zeros((N, N))
    K[0, 0] = Omega ** 2 + np.sum(csec ** 2 / wsec ** 2)
    K[0, 1:] = csec
    K[1:, 0] = csec
    K[1:, 1:] = np.diag(wsec ** 2)
    vals, U = np.linalg.eigh(K)
    if np.any(vals <= 0.0):
        raise RuntimeError("GOA Hessian is not positive definite")
    omega = np.sqrt(vals)
    force = np.zeros(N)
    force[0] = Omega ** 2 * y0
    dq = 2.0 * np.linalg.solve(K, force)
    R = U.T @ dq
    Er = float(0.5 * np.sum(omega ** 2 * R ** 2))
    omega.setflags(write=False)
    R.setflags(write=False)
    return BathModes(omega=omega, R=R, Er=Er)


def normal_modes(p: ModelParams) -> BathModes:
    """Normal modes for a model (cached on the bath parameters)."""
    return _normal_modes(float(p.Omega), float(p.y0), int(p.N), float(p.eta), float(p.omega_c))


def lab_correlation(modes: BathModes, beta: float, tau) -> np.ndarray:
    """C(tau) = <X(tau) X(0)> of the untransformed bath (the TEMPO input)."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    w, cth, wt = modes.omega, thermal_coth(beta, modes.omega), modes.lab_weights
    ph = np.outer(w, tau)
    return (wt * cth) @ np.cos(ph) - 1j * wt @ np.sin(ph)
