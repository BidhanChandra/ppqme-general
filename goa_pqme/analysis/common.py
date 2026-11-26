"""Shared helpers of the paper analyses (previously copy-pasted into every script).

Data layout of a *figures folder* (default goa_figs_v2/, built by `analysis.build frames`):
    csv/<tag>_tempo_T24_dt0.1_eps*.csv     TEMPO references (t, P_D, P_A, dP, Re_rho_DA, Im_rho_DA)
    csv/<tag>_frame_errors.csv             alpha, omega_h, r3, r4, stable_<v>, err_<v>     (v = TCL2, K3, K4, TCL234)
    csv/<tag>_coherence_errors.csv         method, readout, frames, oracle, typical, RF, ...
    curves/<case_id>_<alpha>_<omega_h>.npz t, TCL2, K3, K4, TCL234  (dP on the dt = 0.05 grid)
Definitions:  error = 0.5 rms(dP - dP_TEMPO) on t <= t_end;  stable: finite and max|dP| <= c (c = 1.6);
RF = error of the pointwise median over the stable frames of a method.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..cases import DEV_PREFIXES, HELD_PREFIXES, case_labels, key_of, match_prefix, stem_of, tempo_file  # noqa: F401
from ..config import FIGS_DIR, STABLE_MAX, TEMPO_DIR  # noqa: F401

VARIANTS4 = ("TCL2", "K3", "K4", "TCL234")


def stable(y, c: float = STABLE_MAX) -> bool:
    return bool(np.all(np.isfinite(y)) and np.abs(y).max() <= c)


def rf_error(Y, m, rd, agg=np.median) -> float:
    """0.5 rms(agg over frames - reference) on the mask m; nan if no frame."""
    return np.nan if (Y is None or len(Y) == 0) else 0.5 * np.sqrt(np.mean((agg(np.asarray(Y), 0)[m] - rd[m]) ** 2))


def figure_keys(csv_dir: str) -> list:
    """Tags that have a *_frame_errors.csv in csv_dir."""
    return sorted({os.path.basename(f).split("_frame_errors")[0] for f in glob.glob(os.path.join(csv_dir, "*_frame_errors.csv"))})


def dev_held(keys) -> tuple:
    """(development tags C1..C14, held-out tags H1..H6) in label order."""
    return [match_prefix(p, keys) for p in DEV_PREFIXES], [match_prefix(p, keys) for p in HELD_PREFIXES]


def read_reference(path: str):
    import pandas as pd
    r = pd.read_csv(path)
    if "t" not in r.columns and "t_au" in r.columns:
        r["t"] = r["t_au"]
    return r


def load_curve(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def figure_curves(cur_dir: str, stem: str) -> list:
    """[(alpha, omega_h, {t, TCL2, ...}), ...] from <cur_dir>/<stem>_<alpha>_<omega_h>.npz (glob order)."""
    out = []
    for f in glob.glob(os.path.join(cur_dir, stem + "_*.npz")):
        b = os.path.basename(f)[:-4].split("_")
        out.append((float(b[-2]), float(b[-1]), load_curve(f)))
    return out


def scan_reference(tempo_dir: str, cid: str):
    """Production TEMPO reference of a case id in a folder of tag-named csv files (dt0.1, T24 except G1)."""
    key = key_of(cid)
    fs = sorted(f for f in glob.glob(os.path.join(tempo_dir, key + "_*tempo*")) if "dt0.1" in f and ("_T24_" in f or cid.startswith("G1_")))
    if not fs:
        raise FileNotFoundError("no TEMPO reference for %s in %s" % (cid, tempo_dir))
    return read_reference(fs[0]), fs
