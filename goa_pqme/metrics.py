r"""Errors, stability screen and frame-ensemble statistics.

Frame error against a reference (TEMPO) on the reference grid t_i <= t_max:
    eps = 1/2 sqrt( mean_i [dP(t_i) - dP_ref(t_i)]^2 )        (= RMSE of P_D;  dP linearly interpolated)
Stable frame:  dP finite everywhere and max_t |dP| <= STABLE_MAX (1.6).
Over the stable frames of one method:
    oracle  = min eps,   typical = median eps,   RF (reference-free) = eps of the pointwise median curve.
Transformed-frame positivity:  lambda_min = (1 - sqrt(dP^2 + 4 |rho~_DA|^2)) / 2.
"""
from __future__ import annotations

import numpy as np

from .config import STABLE_MAX


def half_rmse(t: np.ndarray, y: np.ndarray, t_ref: np.ndarray, y_ref: np.ndarray) -> float:
    m = t_ref <= t[-1] + 1e-9
    with np.errstate(all="ignore"):
        e = float(np.sqrt(np.mean((np.interp(t_ref[m], t, y) - y_ref[m]) ** 2)) / 2)
    return e if np.isfinite(e) else float("inf")


def is_stable(y: np.ndarray, threshold: float = STABLE_MAX) -> bool:
    return bool(np.all(np.isfinite(y)) and np.nanmax(np.abs(y)) <= threshold)


def lambda_min(dP: np.ndarray, rho_DA: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.sqrt(dP ** 2 + 4.0 * np.abs(rho_DA) ** 2))


VARIANTS4 = ("TCL2", "K3", "K4", "TCL234")


def frame_table(ref_t: np.ndarray, ref_dP: np.ndarray, curves: dict, variants=VARIANTS4, threshold: float = STABLE_MAX):
    """Per-frame errors (the *_frame_errors.csv layout).

    curves: {(alpha, omega_h): {'t', <variant>: dP, 'r3', 'r4'}}.  Errors on the reference grid t_i <= max t.
    Returns (DataFrame[alpha, omega_h, r3, r4, stable_<v>, err_<v>], interpolated curves {(a, w, v): y}, tt, ref on tt).
    """
    import pandas as pd
    tmax = max(o["t"][-1] for o in curves.values())
    tt = ref_t[ref_t <= tmax + 1e-9]
    rdp = ref_dP[:len(tt)]
    rows, interp = [], {}
    for (a, w), o in curves.items():
        r = dict(alpha=a, omega_h=w, r3=float(o["r3"]), r4=float(o["r4"]))
        for m in variants:
            y = np.interp(tt, o["t"], o[m])
            interp[(a, w, m)] = y
            r["stable_" + m] = is_stable(o[m], threshold)
            e = np.sqrt(np.mean((y - rdp) ** 2)) / 2
            r["err_" + m] = e if np.isfinite(e) else np.inf
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["alpha", "omega_h"]).reset_index(drop=True)
    return df, interp, tt, rdp


def analyse(curves: dict, ref, threshold: float = STABLE_MAX):
    """Frame table + oracle / typical / RF for TCL2 and TCL234 (the notebook comparison).

    curves: {(alpha, omega_h): {'t', 'TCL2', 'K3', 'K4', 'TCL234', 'r3', 'r4'}};  ref: DataFrame with t, dP.
    Returns (df, summary{method: {...}}, tt, interpolated curves {(a, w, method): y}).
    """
    df, interp, tt, rdp = frame_table(ref.t.values, ref.dP.values, curves, threshold=threshold)
    summ = {}
    for name in ("TCL2", "TCL234"):
        ok = df[df["stable_" + name]]
        if len(ok):
            med = np.median([interp[(r.alpha, r.omega_h, name)] for r in ok.itertuples()], 0)
            b = ok.loc[ok["err_" + name].idxmin()]
            summ[name] = dict(n_stable=len(ok), n_frames=len(df), oracle=float(b["err_" + name]), oracle_frame=(b.alpha, b.omega_h),
                              typical=float(ok["err_" + name].median()), RF=float(np.sqrt(np.mean((med - rdp) ** 2)) / 2), median_curve=med)
        else:
            summ[name] = dict(n_stable=0, n_frames=len(df), oracle=np.nan, oracle_frame=None, typical=np.nan, RF=np.nan,
                              median_curve=np.full(len(tt), np.nan))
    return df, summ, tt, interp


def ensemble_summary(curves: dict, t_ref: np.ndarray, y_ref: np.ndarray, t: np.ndarray, threshold: float = STABLE_MAX) -> dict:
    """curves: {frame_key: dP on grid t}.  Returns oracle / typical / RF over the stable frames."""
    m = t_ref <= t[-1] + 1e-9
    tt, yr = t_ref[m], y_ref[m]
    stable = {k: np.interp(tt, t, y) for k, y in curves.items() if is_stable(y, threshold)}
    if not stable:
        return dict(n_stable=0, n_frames=len(curves), oracle=np.nan, oracle_frame=None, typical=np.nan, RF=np.nan)
    errs = {k: float(np.sqrt(np.mean((y - yr) ** 2)) / 2) for k, y in stable.items()}
    best = min(errs, key=errs.get)
    med = np.median(np.array(list(stable.values())), 0)
    return dict(n_stable=len(stable), n_frames=len(curves), oracle=errs[best], oracle_frame=best,
                typical=float(np.median(list(errs.values()))), RF=float(np.sqrt(np.mean((med - yr) ** 2)) / 2),
                median_curve=med, t=tt)
