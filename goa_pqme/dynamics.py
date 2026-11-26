r"""Assemble the full generator, propagate, read out observables.

Full Schroedinger-picture generator on the fine grid (dt):
    L_TCL2   = L                            (TCL2 from tcl2.py, includes -i[H~_S, .])
    L_K3     = L + K3_S(t),   L_K4 = L + K4_S(t),   L_TCL234 = L + K3_S(t) + K4_S(t)
    Kn_S(t)  = CubicSpline_{not-a-knot}( {t_k}, { P(t_k) Kn(t_k) P(t_k)^{-1} } )(t)      (Kn(0) = 0)

Propagation (exponential midpoint rule, second order):
    vec rho_{k+1} = exp[ (L(t_k) + L(t_{k+1})) dt / 2 ] vec rho_k,     rho_0 = U^+ |D><D| U.

Observables:
    dP(t) = rho_DD - rho_AA  (site basis; identical in the lab and the transformed frame),
    rho_DA^lab(t) ~ B rho~_DA(t) + Q1(t)  with
    Q1(t) = (-i/Gamma) int_0^t dtau sum_b { C_{AD,b}(s) Tr[S_AD(s) S_b rho~(tau)] - C^-_{b,AD}(s) Tr[S_AD(s) rho~(tau) S_b] },
    s = t - tau,  [S(s)]_mn = e^{i(E_m-E_n)s} S_mn   (first order in H~_I).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from .frame import PolaronFrame
from .liouville import to_schrodinger

VARIANTS = ("TCL2", "K3", "K4", "TCL234")


def interpolate_generator(kn_t: np.ndarray, K: np.ndarray, E: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rotate interaction-picture K (nt_nodes,4,4) to the Schroedinger picture and spline onto t."""
    Ks = np.zeros_like(K)
    Ks[1:] = to_schrodinger(K[1:], E, kn_t[1:])
    n = len(kn_t)
    return CubicSpline(kn_t, Ks.reshape(n, 16), axis=0)(t).reshape(len(t), 4, 4)


def variant_generators(L: np.ndarray, K3s: np.ndarray, K4s: np.ndarray) -> dict:
    return {"TCL2": L, "K3": L + K3s, "K4": L + K4s, "TCL234": L + K3s + K4s}


def rho0_donor(frame: PolaronFrame) -> np.ndarray:
    r = np.array([[1, 0], [0, 0]], complex)
    return frame.U.conj().T @ r @ frame.U


def propagate(L: np.ndarray, t: np.ndarray, rho0: np.ndarray) -> np.ndarray:
    """Midpoint-exponential propagation of vec rho; returns rho (nt,2,2) in the basis of rho0."""
    dt = np.diff(t)
    Lm = 0.5 * (L[1:] + L[:-1])
    v = rho0.reshape(-1, order="F")
    out = np.zeros((len(t), 4), complex)
    out[0] = v
    for k in range(len(t) - 1):
        v = sla.expm(Lm[k] * dt[k]) @ v
        out[k + 1] = v
    return np.array([o.reshape(2, 2, order="F") for o in out])


def to_site(frame: PolaronFrame, rho_e: np.ndarray) -> np.ndarray:
    U = frame.U
    return np.einsum("ij,tjk,lk->til", U, rho_e, U.conj())


def population_difference(frame: PolaronFrame, rho_e: np.ndarray) -> np.ndarray:
    rs = to_site(frame, rho_e)
    return (rs[:, 0, 0] - rs[:, 1, 1]).real


def q1_kernel(frame: PolaronFrame, t: np.ndarray, C: np.ndarray, Cm: np.ndarray, a: int = 3) -> np.ndarray:
    """c[s,k,m] such that Q1(t_i) = -i dt/Gamma * trapezoid_s sum_km c[s,k,m] rho_km(t_i - s)."""
    E, S = frame.E, frame.S
    SaH = np.exp(1j * (E[:, None] - E[None, :])[None] * t[:, None, None]) * S[a][None]
    c = np.zeros((len(t), 2, 2), complex)
    for b in range(4):
        c += C[a, b][:, None, None] * np.einsum("smn,nk->skm", SaH, S[b]) \
            - Cm[b, a][:, None, None] * np.einsum("smn,km->snk", SaH, S[b])
    return c


def q1(frame: PolaronFrame, t: np.ndarray, C: np.ndarray, Cm: np.ndarray, rho_e: np.ndarray, a: int = 3) -> np.ndarray:
    """First-order readout correction Q1(t) (FFT convolution with trapezoid end corrections)."""
    c = q1_kernel(frame, t, C, Cm, a)
    nt, dt = len(t), t[1] - t[0]
    out = np.zeros(nt, complex)
    for k in range(2):
        for m in range(2):
            out += fftconvolve(c[:, k, m], rho_e[:, k, m])[:nt]
    out -= 0.5 * np.einsum("km,tkm->t", c[0], rho_e) + 0.5 * np.einsum("tkm,km->t", c, rho_e[0])
    return -1j * dt * out / frame.Gamma


@dataclass
class FrameResult:
    """Result of one frame: dP curves of every variant on the fine grid, plus generator diagnostics."""
    alpha: float
    omega_h: float
    t: np.ndarray
    dP: dict
    r3: float
    r4: float
    B: float
    lambda_p: float
    E: np.ndarray
    knodes: object = field(default=None, repr=False)
    coherence: dict = field(default_factory=dict, repr=False)
    seconds: float = 0.0
    U: np.ndarray = field(default=None, repr=False)        # eigenvectors of H~_S (energy -> site)
    rho: dict = field(default_factory=dict, repr=False)    # site density matrices (only if requested)
