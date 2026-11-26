"""Frame-independence summary per case (spread of the frame ensemble and how often TCL234 beats TCL2).

  python -m goa_pqme.analysis.frame_independence --scan work/scan [--out DIR]              (new scan: summary.csv + curves)
  python -m goa_pqme.analysis.frame_independence --legacy work                              (results_full.jsonl + work/curves)

Per case (cases with >= 120 frames): best / median TCL2 error, best / median TCL234 error over frames with max|TCL234| <= 1.6,
stable fraction, ensemble spread  mean_t std_frames(dP)/2  over frames stable for both methods, the same restricted to
'plausibly convergent' frames (r3, r4 < 0.5), and the fraction of TCL234-stable frames where TCL234 beats TCL2.
Writes frame_independence_summary.csv (and full_scan_results.csv for --legacy).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..cases import load_reference
from ..config import WORK
from ..io import iter_frames, load_frame


def summarise(d, curves):
    """d: DataFrame[case, alpha, omega_h, TCL2, TCL234, max_TCL2, max_TCL234, r3, r4];  curves[(case,a,w)] = (TCL2, TCL234) on a common grid."""
    import pandas as pd
    d = d.copy()
    d["stable234"] = d.max_TCL234 <= 1.6; d["stable2"] = d.max_TCL2 <= 1.6; d["gain"] = d.TCL2 - d.TCL234
    rows = []
    for c, g in d.groupby("case"):
        if len(g) < 120:
            continue
        Y2 = np.array([curves[(c, r.alpha, r.omega_h)][0] for r in g.itertuples()]); Y4 = np.array([curves[(c, r.alpha, r.omega_h)][1] for r in g.itertuples()])
        st = g.stable234.values & g.stable2.values
        sp = lambda Y, m: float(np.mean(np.std(Y[m], axis=0) / 2)) if m.sum() > 2 else np.nan
        conv = st & (g["r4"].values < 0.5) & (g["r3"].values < 0.5)
        rows.append(dict(case=c, n=len(g), best2=g.TCL2.min(), best234=g.TCL234[g.stable234].min(), med2=g.TCL2.median(), med234=g.TCL234[g.stable234].median(),
                         frac_stable234=g.stable234.mean(), spread2_all=sp(Y2, st), spread234_all=sp(Y4, st), n_conv=int(conv.sum()),
                         spread2_conv=sp(Y2, conv), spread234_conv=sp(Y4, conv), frac_234_better=float((g.TCL234 < g.TCL2)[g.stable234].mean())))
    return pd.DataFrame(rows), d


def read_legacy(work):
    """Legacy production scan: work/results_full.jsonl (+ work/curves/<cid>_<a>_<w>.npz with t, ref, TCL2, ...)."""
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(work, "results_full.jsonl"))]).drop_duplicates(["case", "alpha", "omega_h"])
    d["r3"], d["r4"] = d["K3rms/K2rms"], d["K4rms/K2rms"]
    curves = {}
    for r in d.itertuples():
        f = os.path.join(work, "curves", "%s_%s_%s.npz" % (r.case, r.alpha, r.omega_h))
        if os.path.exists(f):
            z = np.load(f); curves[(r.case, r.alpha, r.omega_h)] = (z["TCL2"], z["TCL234"])
    return d, curves


def read_scan(scan):
    """New scan folder: errors from TEMPO references (data/trajectories), curves interpolated onto the reference grid."""
    import pandas as pd
    from ..metrics import half_rmse
    rows, curves, refs = [], {}, {}
    for path, cid, a, w in iter_frames(scan):
        if cid not in refs:
            try:
                refs[cid] = load_reference(cid)
            except FileNotFoundError:
                refs[cid] = None
        ref = refs[cid]
        if ref is None:
            continue
        z = load_frame(path); t = z["t"]; tr = ref.t.values[ref.t.values <= t[-1] + 1e-9]
        r = dict(case=cid, alpha=a, omega_h=w, r3=float(z["r3"]), r4=float(z["r4"]))
        for m in ("TCL2", "TCL234"):
            y = z["dP"][m]
            r[m] = half_rmse(t, y, ref.t.values, ref.dP.values)
            r["max_" + m] = float(np.nanmax(np.abs(y))) if np.all(np.isfinite(y)) else float("inf")
        rows.append(r)
        with np.errstate(all="ignore"):
            curves[(cid, a, w)] = (np.interp(tr, t, z["dP"]["TCL2"]), np.interp(tr, t, z["dP"]["TCL234"]))
    return pd.DataFrame(rows), curves


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--scan"); g.add_argument("--legacy")
    ap.add_argument("--out", default=None); a = ap.parse_args(argv)
    d, curves = read_legacy(a.legacy) if a.legacy else read_scan(a.scan)
    out = a.out or (a.legacy or a.scan); os.makedirs(out, exist_ok=True)
    R, full = summarise(d, curves)
    pd.set_option("display.width", 250); print(R.round(4).to_string(index=False))
    R.to_csv(os.path.join(out, "frame_independence_summary.csv"), index=False)
    if a.legacy:
        full.drop(columns=["r3", "r4"]).to_csv(os.path.join(out, "full_scan_results.csv"), index=False)


if __name__ == "__main__":
    main()
