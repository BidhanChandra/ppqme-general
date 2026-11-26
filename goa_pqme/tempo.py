r"""Numerically exact reference with OQuPy TEMPO (optional dependency).

Same discrete bath and initial state as the TCL calculations:
    system   H_S = [[omega_DA, Gamma], [Gamma, E_r]],   coupling operator |A><A|,
    C(tau)   = sum_q (w_q^3 R_q^2 / 2) [coth(beta w_q/2) cos(w_q tau) - i sin(w_q tau)],
    rho(0)   = |D><D|.
Production settings: dt = 0.1, epsrel = 1e-8 (beta = inf) or 1e-9 (finite beta), tmax = 24.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from .bath import lab_correlation, normal_modes
from .config import ModelParams
from .frame import lab_hamiltonian


def production_epsrel(beta: float) -> float:
    return 1e-8 if math.isinf(beta) else 1e-9


@contextlib.contextmanager
def robust_svd():
    """Temporarily make np.linalg.svd fall back to LAPACK gesvd (then a 1e-14 jitter) when gesdd fails."""
    import scipy.linalg as sl
    orig = np.linalg.svd

    def safe(a, full_matrices=True, compute_uv=True, hermitian=False):
        try:
            return orig(a, full_matrices, compute_uv, hermitian)
        except np.linalg.LinAlgError:
            try:
                return sl.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesvd", check_finite=False)
            except Exception:
                rng = np.random.default_rng(0)
                a2 = a + np.max(np.abs(a)) * 1e-14 * rng.standard_normal(a.shape)
                return sl.svd(a2, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesvd", check_finite=False)

    np.linalg.svd = safe
    try:
        yield
    finally:
        np.linalg.svd = orig


def run_tempo(p: ModelParams, tmax: float = 24.0, dt: float = 0.1, epsrel: float | None = None, tcut=None, robust: bool = True):
    """Returns a DataFrame with columns t, P_D, P_A, dP, Re_rho_DA, Im_rho_DA."""
    import oqupy
    import pandas as pd
    modes = normal_modes(p)
    eps = production_epsrel(p.beta) if epsrel is None else epsrel
    corr = oqupy.CustomCorrelations(lambda tau: complex(lab_correlation(modes, p.beta, float(tau))[0]))
    bath = oqupy.Bath(np.array([[0, 0], [0, 1]], complex), corr)
    par = oqupy.TempoParameters(dt=dt, epsrel=eps, tcut=tcut)
    ctx = robust_svd() if robust else contextlib.nullcontext()
    with ctx:
        dyn = oqupy.tempo_compute(system=oqupy.System(lab_hamiltonian(p)), bath=bath,
                                  initial_state=np.array([[1, 0], [0, 0]], complex),
                                  start_time=0.0, end_time=tmax, parameters=par, progress_type="silent")
    t, rho = dyn.times, np.array(dyn.states)
    return pd.DataFrame(dict(t=t, P_D=rho[:, 0, 0].real, P_A=rho[:, 1, 1].real, dP=(rho[:, 0, 0] - rho[:, 1, 1]).real,
                             Re_rho_DA=rho[:, 0, 1].real, Im_rho_DA=rho[:, 0, 1].imag))
