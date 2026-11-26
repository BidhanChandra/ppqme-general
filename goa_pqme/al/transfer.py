r"""Cross-case transfer of the dK surrogate (few-shot).

Source set: n_src random frames from every other case; features = normalised (log alpha, log omega_h, B, lambda_p, ||K2||)
plus normalised case descriptors (log Gamma, omega_DA, 1/beta, y0).  A PCA+GP model (nc = 8) trained on the sources plus
m target frames (m in fewshots) predicts the remaining target frames; optionally Y is divided by ||K2|| (--scaled).
Reports generator errors (aggregate, median) and population errors as in pop_eval.

  python -m goa_pqme.al.transfer CASE [CASE ...] --scan work/scan_production [--scaled] [--cache work/al/cache.npy]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..cases import get_case, load_reference
from ..io import frame_filename, load_frame
from ..metrics import half_rmse
from .data import CACHE, load_cache
from .pop_eval import pop_curves
from .surrogate import fit, predict


def _features(cache):
    cases = list(cache)

    def cf(c):
        p = get_case(c)
        return [np.log(p.Gamma), p.omega_DA, 0.0 if np.isinf(p.beta) else 1.0 / p.beta, p.y0]
    CF = np.array([cf(c) for c in cases]); CF = (CF - CF.min(0)) / (CF.max(0) - CF.min(0) + 1e-12)
    allX = np.vstack([np.c_[np.log(cache[c]["X"]), cache[c]["F"]] for c in cases]); lo, hi = allX.min(0), allX.max(0)

    def feat(c):
        a = np.c_[np.log(cache[c]["X"]), cache[c]["F"]]; a = (a - lo) / (hi - lo + 1e-12)
        return np.c_[a, np.tile(CF[cases.index(c)], (len(a), 1))]
    return cases, feat


def run(target, cache, scan, n_src=30, fewshots=(0, 10, 20), seed=0, scaled=False):
    cases, feat = _features(cache)
    sc = lambda c: (cache[c]["F"][:, 2:3] if scaled else 1.0)
    rng = np.random.default_rng(seed); d = cache[target]; ref = load_reference(target); p = get_case(target)
    tn = d.get("t_nodes", np.arange(0, 24 + 1e-9, 3.0)); N = len(d["Y"])
    Ys, Xs = [], []
    for c in cases:
        if c == target:
            continue
        idx = rng.choice(len(cache[c]["Y"]), n_src, replace=False); Ys.append((cache[c]["Y"] / sc(c))[idx]); Xs.append(feat(c)[idx])
    Ys = np.vstack(Ys); Xs = np.vstack(Xs); Xt = feat(target); rows = []
    for m in fewshots:
        sel = list(rng.choice(N, m, replace=False)) if m else []
        Y = np.vstack([Ys] + ([(d["Y"] / sc(target))[sel]] if m else [])); X = np.vstack([Xs] + ([Xt[sel]] if m else []))
        mod = fit(Y, X, list(range(len(Y))), nc=8)
        te = np.array([i for i in range(N) if i not in sel]); Yh = predict(mod, Xt, te) * (sc(target)[te] if scaled else 1.0); Yt = d["Y"]
        gen = np.linalg.norm(Yh - Yt[te]) / np.linalg.norm(Yt[te]); gmed = np.median(np.linalg.norm(Yh - Yt[te], axis=1) / np.linalg.norm(Yt[te], axis=1))
        Nn = Yh.shape[1] // 2; dK = Yh[:, :Nn] + 1j * Yh[:, Nn:]
        for ii, i in enumerate(te):
            a, w = d["X"][i]; z = load_frame(frame_filename(scan, target, a, w)); t, dP = pop_curves(p, a, w, dK[ii], tn)
            e = lambda tt, v: half_rmse(tt, v, ref.t.values, ref.dP.values) if np.all(np.isfinite(v)) else np.inf
            tr = ref.t.values[ref.t.values <= t[-1] + 1e-9]; y, ytrue = np.interp(tr, t, dP), np.interp(tr, z["t"], z["dP"]["TCL234"]); lab = d["lab"][i]
            rows.append(dict(scaled=scaled, target=target, n_target=m, n_src=n_src, agg=gen, med=gmed, alpha=a, omega_h=w,
                             rm_tcl2=e(z["t"], z["dP"]["TCL2"]), rm_true234=e(z["t"], z["dP"]["TCL234"]), rm_pred=e(t, dP),
                             unst_true=bool(lab[3] > 1.6 or not lab[2]), unst_pred=bool(np.nanmax(np.abs(y)) > 1.6 or not np.all(np.isfinite(y))),
                             dev=float(np.sqrt(np.mean((np.clip(y, -2, 2) - np.clip(ytrue, -2, 2)) ** 2)) / 2)))
        print(target, m, "gen agg %.3f med %.3f" % (gen, gmed), flush=True)
    return rows


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+"); ap.add_argument("--scan", required=True); ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--scaled", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cache = load_cache(a.cache)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cache)), "cross_eval%s.csv" % ("_scaled" if a.scaled else ""))
    rows = []
    for t in a.cases:
        rows += run(t, cache, a.scan, scaled=a.scaled); pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
