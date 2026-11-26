"""Robustness of the reference-free (RF) ensemble result, from the per-frame curves of a figures folder (no new dynamics).

  python -m goa_pqme.analysis.ensembles threshold [--data goa_figs_v2] [--cs 1.6,1.3,1.1,1.0]   -> <data>/sens_threshold.csv
  python -m goa_pqme.analysis.ensembles checks    [--data goa_figs_v2]

threshold: RF errors when the stability screen |dP| <= c is tightened (c = 1.6 paper; physical populations need c = 1).
checks:    (a) same-frame medians (frames stable for BOTH methods) vs method-specific sets,
           (b) aggregator: median vs 10%-trimmed mean vs mean,
           (c) sub-grids of every second alpha and omega_h.
Errors: 0.5 rms(dP_RF - dP_TEMPO); reported as mean over development / held-out / all, TCL2 -> TCL234.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .common import FIGS_DIR, dev_held, figure_keys, read_reference, stem_of, tempo_file


def threshold(data, cs):
    import pandas as pd
    CSV, CUR = os.path.join(data, "csv"), os.path.join(data, "curves")
    dev, held = dev_held(figure_keys(CSV))
    D = {}
    for k in dev + held:
        ref = read_reference(tempo_file(CSV, k)); fr = []
        for f in glob.glob(os.path.join(CUR, stem_of(k) + "_*.npz")):
            z = np.load(f); fr.append(dict(t=z["t"], TCL2=z["TCL2"], TCL234=z["TCL234"]))
        t = fr[0]["t"]; m = t <= ref.t.iloc[-1] + 1e-9; D[k] = (t, m, np.interp(t, ref.t, ref.dP), fr)

    def ok(y, c):
        return bool(np.all(np.isfinite(y)) and np.abs(y).max() <= c)
    rows = []
    for c in cs:
        r = dict(c=c); res = {}
        for k, (t, m, rd, fr) in D.items():
            o = {}
            for v in ("TCL2", "TCL234"):
                Y = [x[v] for x in fr if ok(x[v], c)]; o[v + "_frac"] = len(Y) / len(fr)
                o[v] = 0.5 * np.sqrt(np.mean((np.median(Y, 0)[m] - rd[m]) ** 2)) if Y else np.nan
            res[k] = o
        R = pd.DataFrame(res).T
        for nm, ks in (("dev", dev), ("held", held), ("all", dev + held)):
            r[nm + "_TCL2"] = R.loc[ks, "TCL2"].mean(); r[nm + "_TCL234"] = R.loc[ks, "TCL234"].mean()
        r["frac_TCL2"] = R.loc[dev, "TCL2_frac"].mean(); r["frac_TCL234"] = R.loc[dev, "TCL234_frac"].mean()
        r["cases_without_stable_TCL234"] = int(R.TCL234.isna().sum()); r["TCL234_worse"] = int((R.TCL234 > R.TCL2).sum()); rows.append(r)
    T = pd.DataFrame(rows).set_index("c"); pd.set_option("display.width", 250); print(T.round(4).to_string())
    T.round(5).to_csv(os.path.join(data, "sens_threshold.csv"))


def checks(data):
    import pandas as pd
    from scipy.stats import trim_mean
    CSV, CUR = os.path.join(data, "csv"), os.path.join(data, "curves")
    dev, held = dev_held(figure_keys(CSV))
    C = 1.6
    stable = lambda y: bool(np.all(np.isfinite(y)) and np.abs(y).max() <= C)
    D = {}
    for k in dev + held:
        ref = read_reference(tempo_file(CSV, k)); fr = []
        for f in glob.glob(os.path.join(CUR, stem_of(k) + "_*.npz")):
            b = os.path.basename(f)[:-4].split("_"); z = np.load(f); fr.append((float(b[-2]), float(b[-1]), z["TCL2"], z["TCL234"]))
        t = np.load(f)["t"]; m = t <= ref.t.iloc[-1] + 1e-9; D[k] = (m, np.interp(t, ref.t, ref.dP), fr)

    def err(Y, m, rd, agg):
        return np.nan if not Y else 0.5 * np.sqrt(np.mean((agg(np.array(Y), 0)[m] - rd[m]) ** 2))
    med, tm, mn = np.median, (lambda a, axis: trim_mean(a, 0.1, axis=axis)), np.mean

    def table(name, fn):
        R = pd.DataFrame({k: fn(*D[k]) for k in D}).T
        out = []
        for nm, ks in (("dev", dev), ("held", held), ("all", dev + held)):
            a, b = R.loc[ks, "TCL2"], R.loc[ks, "TCL234"]
            out.append("%s %.4f -> %.4f (%+.0f%%, better in %d/%d)" % (nm, a.mean(), b.mean(), 100 * (b.mean() / a.mean() - 1), (b < a).sum(), len(ks)))
        print("%-34s" % name, " | ".join(out)); return R

    def std(agg, sel=lambda a, w: True):
        def fn(m, rd, fr):
            return dict(TCL2=err([x[2] for x in fr if stable(x[2]) and sel(x[0], x[1])], m, rd, agg),
                        TCL234=err([x[3] for x in fr if stable(x[3]) and sel(x[0], x[1])], m, rd, agg))
        return fn

    def common(m, rd, fr):
        both = [x for x in fr if stable(x[2]) and stable(x[3])]
        return dict(TCL2=err([x[2] for x in both], m, rd, med), TCL234=err([x[3] for x in both], m, rd, med))
    print("--- (a) same frame set vs method-specific sets (median) ---")
    table("method-specific stable frames", std(med)); table("same frames (stable for both)", common)
    print("   frames used: median fraction of the grid in the common set = %.2f" % np.median([sum(stable(x[2]) and stable(x[3]) for x in D[k][2]) / len(D[k][2]) for k in D]))
    print("--- (b) aggregator ---"); table("median", std(med)); table("10% trimmed mean", std(tm)); table("mean", std(mn))
    print("--- (c) sub-grids (every 2nd alpha x every 2nd omega_h; 4 offsets), median ---")
    al = sorted({x[0] for x in D[dev[0]][2]}); wh = sorted({x[1] for x in D[dev[0]][2]})
    for ia in (0, 1):
        for iw in (0, 1):
            sa, sw = set(al[ia::2]), set(wh[iw::2]); table("alpha offset %d, omega_h offset %d" % (ia, iw), std(med, lambda a, w: a in sa and w in sw))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["threshold", "checks"]); ap.add_argument("--data", default=FIGS_DIR); ap.add_argument("--cs", default="1.6,1.3,1.1,1.0")
    a = ap.parse_args(argv)
    threshold(a.data, [float(x) for x in a.cs.split(",")]) if a.what == "threshold" else checks(a.data)


if __name__ == "__main__":
    main()
