r"""Population-level evaluation of predicted generators.

For a predicted site-basis correction dK_site at the nodes, the population dynamics are recomputed exactly as in a scan:
    dK_energy = W^{-1} dK_site W,   spline onto t (dt = 0.05),   L = L_TCL2 + dK_energy,   propagate,  dP = rho_DD - rho_AA.
Errors 0.5 RMSE against TEMPO for TCL2, the true TCL234 (from the scan) and the predicted TCL234.

  python -m goa_pqme.al.pop_eval CASE --scan work/scan_production [--cache work/al/cache.npy]   -> work/al/pop_eval_<CASE>.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.interpolate import CubicSpline

from ..cases import get_case, load_reference
from ..dynamics import population_difference, propagate, rho0_donor
from ..frame import make_frame
from ..io import frame_filename, load_frame
from ..metrics import half_rmse
from ..tcl2 import TCL2
from .data import CACHE, load_cache
from .surrogate import acquire, feats, fit, predict


def pop_curves(params, a, w, dK_site_nodes, t_nodes, dt=0.05):
    """dP(t) of TCL2 + a site-basis correction given at the non-zero nodes ((n-1)*16 complex numbers or (n-1,4,4))."""
    fr = make_frame(params, a, w)
    t = np.arange(0, t_nodes[-1] + 1e-9, dt)
    L = TCL2(fr, t).L
    W = np.kron(fr.U.conj(), fr.U); Wi = np.linalg.inv(W)
    n = len(t_nodes); dKe = np.zeros((n, 4, 4), complex)
    dKe[1:] = np.einsum("ij,tjk,kl->til", Wi, np.asarray(dK_site_nodes).reshape(n - 1, 4, 4), W)
    S = CubicSpline(t_nodes, dKe.reshape(n, 16), axis=0)(t).reshape(len(t), 4, 4)
    with np.errstate(all="ignore"):
        dP = population_difference(fr, propagate(L + S, t, rho0_donor(fr)))
    return t, dP


def evaluate(cid, scan, cache, strategies=(("random", "F0"), ("gp_u", "F3")), seeds=(0,), sizes=(20, 40)):
    import pandas as pd
    d = cache[cid]; p = get_case(cid); ref = load_reference(cid); tn = d.get("t_nodes", np.arange(0, 24 + 1e-9, 3.0))
    rows = []
    for strat, kind in strategies:
        for seed in seeds:
            for nacq in sizes:
                acq = acquire(d, strat, kind, seed, nacq); m = fit(d["Y"], feats(d, kind), acq)
                te = np.array([i for i in range(len(d["Y"])) if i not in acq])
                Yh = predict(m, feats(d, kind), te); Nn = Yh.shape[1] // 2; dK = Yh[:, :Nn] + 1j * Yh[:, Nn:]
                for ii, i in enumerate(te):
                    a, w = d["X"][i]; z = load_frame(frame_filename(scan, cid, a, w)); t, dP = pop_curves(p, a, w, dK[ii], tn)
                    e = lambda tt, v: half_rmse(tt, v, ref.t.values, ref.dP.values) if np.all(np.isfinite(v)) else np.inf
                    m_ = ref.t.values <= t[-1] + 1e-9; tr = ref.t.values[m_]
                    y, ytrue = np.interp(tr, t, dP), np.interp(tr, z["t"], z["dP"]["TCL234"])
                    lab = d["lab"][i]
                    rows.append(dict(case=cid, strategy=strat, feat=kind, seed=seed, n=nacq, alpha=a, omega_h=w,
                                     rm_tcl2=e(z["t"], z["dP"]["TCL2"]), rm_true234=e(z["t"], z["dP"]["TCL234"]), rm_pred=e(t, dP),
                                     unst_true=bool(lab[3] > 1.6 or not lab[2]), unst_pred=bool(np.nanmax(np.abs(y)) > 1.6 or not np.all(np.isfinite(y))),
                                     dev_pred_vs_true=float(np.sqrt(np.mean((np.clip(y, -2, 2) - np.clip(ytrue, -2, 2)) ** 2)) / 2)))
                print(cid, strat, seed, nacq, flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case"); ap.add_argument("--scan", required=True); ap.add_argument("--cache", default=CACHE); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = evaluate(a.case, a.scan, load_cache(a.cache))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cache)), "pop_eval_%s.csv" % a.case)
    df.to_csv(out, index=False); print("written", out)


if __name__ == "__main__":
    main()
