r"""Partial-polaron frame (alpha, omega_h): filter, renormalisation, residual bath, system operators.

Filter (fraction of each mode that is displaced by the polaron transformation):
    f_q = 1 - exp[-(w_q/omega_h)^alpha]            (f -> 1 full polaron, f -> 0 lab frame)

Residual-bath amplitudes (second-quantised, b_q bath annihilators):
    xi_q = sqrt(w_q/2)   R_q f_q          (displacement / hopping part)
    x_q  = sqrt(w_q^3/2) R_q (1 - f_q)    (residual linear coupling)
    c_q  = coth(beta w_q/2) = 2 n_q + 1

Renormalisation:
    B        = exp(-phi(0)/2),   phi(0) = sum_q xi_q^2 c_q
    lambda_p = 1/2 sum_q w_q^2 R_q^2 f_q (2 - f_q)
    H~_S     = [[omega_DA, Gamma B], [Gamma B, E_r - lambda_p]] = U diag(E_0, E_1) U^dagger

Residual interaction  H~_I = sum_{a in AA, DA, AD} S_a (x) B_a :
    S_AA = |A><A|,  B_AA = X~ = sum_q x_q (b_q^+ + b_q)
    S_DA = |D><A|,  B_DA = Gamma (D_-  - B)
    S_AD = |A><D|,  B_AD = Gamma (D_+  - B),     D_s = exp[s sum_q xi_q (b_q^+ - b_q)],  <D_s> = B
so that <B_a> = 0 (P L P = 0).

The three primitive (two-time) bath functions, used by TCL2 and by the coherence readout:
    phi(t)    = sum_q xi_q^2      [c_q cos w t - i sin w t]        (displacement-displacement, DD)
    mixed(t)  = sum_q xi_q x_q    [cos w t - i c_q sin w t]        (displacement-linear,       DX)
    linear(t) = sum_q x_q^2       [c_q cos w t - i sin w t]        (linear-linear,             XX)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bath import BathModes, normal_modes, thermal_coth
from .config import CHANNELS, ModelParams


def polaron_filter(omega, alpha: float, omega_h: float) -> np.ndarray:
    """f(w) = 1 - exp[-(w/omega_h)^alpha]."""
    x = np.maximum(np.asarray(omega, dtype=float) / omega_h, 0.0)
    return -np.expm1(-(x ** alpha))


def system_operators(U: np.ndarray) -> np.ndarray:
    """S_a = U^dagger |j><k| U for (j,k) in CHANNELS (DD, AA, DA, AD): array (4,2,2), energy basis."""
    ops = np.zeros((4, 2, 2), complex)
    for a, (j, k) in enumerate(CHANNELS):
        op = np.zeros((2, 2), complex)
        op[j, k] = 1.0
        ops[a] = U.conj().T @ op @ U
    return ops


@dataclass(frozen=True)
class PolaronFrame:
    """Everything that depends only on (model, alpha, omega_h) -- computed once."""
    params: ModelParams
    alpha: float
    omega_h: float
    modes: BathModes
    f: np.ndarray = field(repr=False)        # (N,) filter values f_q
    coth: np.ndarray = field(repr=False)     # (N,) c_q = coth(beta w_q / 2)
    xi: np.ndarray = field(repr=False)       # (N,) displacement amplitudes
    x: np.ndarray = field(repr=False)        # (N,) residual linear amplitudes
    B: float                                 # <D_s>, coupling renormalisation
    lambda_p: float                          # polaron shift
    H_tilde: np.ndarray = field(repr=False)  # (2,2) transformed system Hamiltonian (site basis)
    E: np.ndarray = field(repr=False)        # (2,) eigenvalues, ascending
    U: np.ndarray = field(repr=False)        # (2,2) eigenvectors (columns), H~ = U diag(E) U^+
    S: np.ndarray = field(repr=False)        # (4,2,2) system operators in the energy basis

    @property
    def omega(self) -> np.ndarray:
        return self.modes.omega

    @property
    def Gamma(self) -> float:
        return self.params.Gamma

    @property
    def nplus_half(self) -> np.ndarray:
        """n_q + 1/2 = c_q / 2."""
        return 0.5 * self.coth

    # ---- primitive bath functions (same operation order as the original goa_user.bath_functions)
    def _sum(self, w_re, w_im, t):
        ph = np.outer(self.omega, np.atleast_1d(np.asarray(t, dtype=float)))
        return w_re @ np.cos(ph) - 1j * (w_im @ np.sin(ph))

    def phi(self, t) -> np.ndarray:
        w, R, f, c = self.omega, self.modes.R, self.f, self.coth
        wt = 0.5 * w * R ** 2 * f ** 2
        return self._sum(wt * c, wt, t)

    def mixed(self, t) -> np.ndarray:
        w, R, f, c = self.omega, self.modes.R, self.f, self.coth
        wt = 0.5 * w ** 2 * R ** 2 * f * (1.0 - f)
        return self._sum(wt, wt * c, t)

    def linear(self, t) -> np.ndarray:
        w, R, f, c = self.omega, self.modes.R, self.f, self.coth
        wt = 0.5 * w ** 3 * R ** 2 * (1.0 - f) ** 2
        return self._sum(wt * c, wt, t)

    def bath_functions(self, t) -> dict:
        return dict(phi=self.phi(t), mixed=self.mixed(t), linear=self.linear(t), B=self.B, lambda_p=self.lambda_p, f=self.f)


def make_frame(p: ModelParams, alpha: float, omega_h: float) -> PolaronFrame:
    """Build the partial-polaron frame for model p and filter (alpha, omega_h)."""
    modes = normal_modes(p)
    w, R = modes.omega, modes.R
    f = polaron_filter(w, alpha, omega_h)
    c = thermal_coth(p.beta, w)
    xi = np.sqrt(w / 2.0) * R * f
    x = np.sqrt(w ** 3 / 2.0) * R * (1.0 - f)
    phi0 = float(0.5 * np.sum(w * R ** 2 * f ** 2 * c))
    B = float(np.exp(-0.5 * phi0))
    lp = float(0.5 * np.sum(w ** 2 * R ** 2 * f * (2.0 - f)))
    Ht = np.array([[p.omega_DA, p.Gamma * B], [p.Gamma * B, modes.Er - lp]], complex)
    E, U = np.linalg.eigh(Ht)
    return PolaronFrame(params=p, alpha=float(alpha), omega_h=float(omega_h), modes=modes, f=f, coth=c, xi=xi, x=x,
                        B=B, lambda_p=lp, H_tilde=Ht, E=E, U=U, S=system_operators(U))


def lab_hamiltonian(p: ModelParams) -> np.ndarray:
    """H_S of the untransformed model, basis (D, A)."""
    return np.array([[p.omega_DA, p.Gamma], [p.Gamma, normal_modes(p).Er]], complex)
