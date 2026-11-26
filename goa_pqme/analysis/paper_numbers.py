"""Numbers quoted in the GOA results text, from the frame-error CSVs and per-frame curves of a figures folder.

  python -m goa_pqme.analysis.paper_numbers [--data goa_figs_v2]      -> <data>/paper_numbers_per_case.csv

Per case: stable fractions, oracle / typical errors (frames stable for the variant), typical error on the frames stable for
all four truncations, median r4 over stable / divergent TCL234 frames, and the RF errors of TCL2 and TCL234.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .common import FIGS_DIR, VARIANTS4, case_labels, figure_keys, read_reference, stem_of, tempo_file


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=FIGS_DIR); A = ap.parse_args(argv)
    CSV, CUR = os.path.join(A.data, "csv"), os.path.join(A.data, "curves")
    keys = figure_keys(CSV)
    LAB = case_labels(keys)
    order = sorted(LAB, key=lambda k: (LAB[k][0] == "H", int(LAB[k][1:])))
    dev = [k for k in order if LAB[k][0] == "C"]; held = [k for k in order if LAB[k][0] == "H"]
    V = list(VARIANTS4)
    FE = {k: pd.read_csv(os.path.join(CSV, k + "_frame_errors.csv")) for k in order}
    R = {}
    for k in order:
        fe = FE[k]; ref = read_reference(tempo_file(CSV, k)); stem = stem_of(k); d = dict()
        for v in V:
            s = fe[fe["stable_" + v].astype(bool) & np.isfinite(fe["err_" + v])]
            d[v + "_stable"] = fe["stable_" + v].astype(bool).mean(); d[v + "_oracle"] = s["err_" + v].min(); d[v + "_typ"] = s["err_" + v].median()
        allst = fe[fe[["stable_" + v for v in V]].astype(bool).all(axis=1)]
        for v in V:
            d[v + "_typ_common"] = allst["err_" + v].median() if len(allst) else np.nan
        d["n_common"] = len(allst)
        st = fe.stable_TCL234.astype(bool); d["r4_stable"] = fe.r4[st].median() if st.any() else np.nan; d["r4_div"] = fe.r4[~st].median() if (~st).any() else np.nan
        stack = {"TCL2": [], "TCL234": []}; t = None
        for r in fe.itertuples():
            fn = os.path.join(CUR, "%s_%s_%s.npz" % (stem, float(r.alpha), float(r.omega_h)))
            if not os.path.exists(fn):
                continue
            z = np.load(fn); t = z["t"]
            for v in stack:
                if bool(getattr(r, "stable_" + v)) and np.all(np.isfinite(z[v])):
                    stack[v].append(z[v])
        if t is not None:
            rd = np.interp(t, ref.t, ref.dP); m = t <= ref.t.iloc[-1] + 1e-9
            for v, x in stack.items():
                if x:
                    d["RF_" + v] = 0.5 * np.sqrt(np.mean((np.median(x, 0)[m] - rd[m]) ** 2))
        R[k] = d
    T = pd.DataFrame(R).T; T.index = [LAB[k] for k in T.index]; pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    dl = [LAB[k] for k in dev]; hl = [LAB[k] for k in held]
    print("=== per case ===\n", T[["RF_TCL2", "RF_TCL234", "TCL2_oracle", "TCL234_oracle", "TCL2_typ", "TCL234_typ", "TCL234_stable", "r4_stable", "r4_div"]].astype(float).round(4).to_string())

    def block(name, a, b):
        x, y = T[a].astype(float), T[b].astype(float)
        print(f"{name:8s} all {x.mean():.4f}->{y.mean():.4f} | dev {x[dl].mean():.4f}->{y[dl].mean():.4f} ({100*(y[dl].mean()/x[dl].mean()-1):+.0f}%) | held {x[hl].mean():.4f}->{y[hl].mean():.4f} ({100*(y[hl].mean()/x[hl].mean()-1):+.0f}%) | better all {int((y<x).sum())}/{len(x)} dev {int((y[dl]<x[dl]).sum())}/{len(dl)} held {int((y[hl]<x[hl]).sum())}/{len(hl)} | worse>10%: {list(x.index[y>1.1*x])}")
    print("\n=== error summaries (TCL2 -> TCL234) ==="); block("RF", "RF_TCL2", "RF_TCL234"); block("oracle", "TCL2_oracle", "TCL234_oracle"); block("typical", "TCL2_typ", "TCL234_typ")
    print("\n=== attribution (frames stable in all four truncations; typical error, mean over development cases) ===")
    for v in V:
        print(f"{v:7s} {T.loc[dl, v + '_typ_common'].astype(float).mean():.4f}   better than TCL2 in {int((T.loc[dl, v + '_typ_common'].astype(float) < T.loc[dl, 'TCL2_typ_common'].astype(float)).sum())} of {len(dl)}")
    print("stable fraction, mean over development cases:", {v: round(float(T.loc[dl, v + '_stable'].astype(float).mean()), 2) for v in V})
    print("\n=== TCL234 stable fraction per case ===\n", T["TCL234_stable"].astype(float).round(2).to_dict())
    r4s, r4d = T.r4_stable.astype(float), T.r4_div.astype(float)
    big = [c for c in ("C8", "H5", "H6") if c in T.index]; rest = [l for l in T.index if l not in big]
    print("\n=== r4 (median over frames, per case) ===")
    print("divergent frames: %.2f-%.2f (cases other than %s); %.2f-%.2f (%s)" % (r4d[rest].min(), r4d[rest].max(), big, r4d[big].min(), r4d[big].max(), big))
    print("stable frames:    %.2f-%.2f over all cases (H3 = %.2f at beta=2, omega_DA=0; maximum case %s)" % (r4s.min(), r4s.max(), r4s.get("H3", np.nan), r4s.idxmax()))
    print("\n=== resonant cases (TCL234 worse than TCL2) ===")
    for nm in ("RF", "typical", "oracle"):
        a, b = {"RF": ("RF_TCL2", "RF_TCL234"), "typical": ("TCL2_typ", "TCL234_typ"), "oracle": ("TCL2_oracle", "TCL234_oracle")}[nm]
        print(nm, [(l, round(float(T.loc[l, a]), 4), round(float(T.loc[l, b]), 4)) for l in T.index if float(T.loc[l, b]) > float(T.loc[l, a])])
    path = os.path.join(A.data, "paper_numbers_per_case.csv")
    T.round(5).to_csv(path); print("\nwritten", path)


if __name__ == "__main__":
    main()
