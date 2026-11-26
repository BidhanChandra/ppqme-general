"""Comparison with Lai & Geva (digitised MCTDH curves) using the per-frame curves of a figures folder.
Their TCL/TC columns are taken unchanged from lg_comparison.csv; ours (dressed frame omega_h = 0.01, RF median, TEMPO)
are recomputed.  Errors are 0.5 rms(sigma_z - MCTDH) on the digitised Gamma*t grid.

  python -m goa_pqme.analysis.lg_compare [--data goa_figs_v2] [--digitised .../digitised_laigeva_curves.csv] [--old .../lg_comparison.csv]
  -> <data>/lg_recomputed.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .common import FIGS_DIR, read_reference, stem_of, tempo_file

PANELS = {"w0_Binf": "G0p2_OD0_Binf_y0p75", "wEr_Binf": "G0p2_OD0p28125_Binf_y0p75", "w0p1_Binf": "G0p2_OD0p1_Binf_y0p75",
          "w0p5_Binf": "G0p2_OD0p5_Binf_y0p75", "w0p2_Binf": "G0p2_OD0p2_Binf_y0p75", "w1_Binf": "G0p2_OD1_Binf_y0p75",
          "w0_B5": "G0p2_OD0_B5_y0p75", "wEr_B5": "G0p2_OD0p28125_B5_y0p75", "w0_B1": "G0p2_OD0_B1_y0p75",
          "wEr_B1": "G0p2_OD0p28125_B1_y0p75", "y1p5_B1": "G0p2_OD0_B1_y1p5", "y0p25_B1": "G0p2_OD0_B1_y0p25"}


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=FIGS_DIR); ap.add_argument("--digitised", default=None); ap.add_argument("--old", default=None)
    A = ap.parse_args(argv)
    dig = A.digitised or os.path.join(A.data, "lg_data", "digitised_laigeva_curves.csv")
    oldf = A.old or os.path.join(A.data, "lg_data", "lg_comparison.csv")
    D = pd.read_csv(dig); old = pd.read_csv(oldf).set_index("case")
    csv_dir = os.path.join(A.data, "csv")
    keys = {stem_of(os.path.basename(f).split("_tempo")[0]): os.path.basename(f).split("_tempo")[0] for f in glob.glob(os.path.join(csv_dir, "*_tempo_T*_dt0.1_*.csv"))}
    st = lambda y: bool(np.all(np.isfinite(y)) and np.abs(y).max() <= 1.6)
    rows = []
    for panel, stem in PANELS.items():
        mm = D[(D.panel == panel) & (D.series == "mctdh")].sort_values("Gamma_t"); xm, ym = mm.Gamma_t.values, mm.sigma_z.values
        fr = []
        for f in glob.glob(os.path.join(A.data, "curves", stem + "_*.npz")):
            b = os.path.basename(f)[:-4].split("_"); fr.append((float(b[-2]), float(b[-1]), np.load(f)))
        if not fr or stem not in keys:
            print("missing", panel); continue
        t = fr[0][2]["t"]; g = 0.2 * t; ref = read_reference(tempo_file(csv_dir, keys[stem])); tem = np.interp(t, ref.t, ref.dP)

        def med(m, sel=lambda w: True):
            Y = [z[m] for (a, w, z) in fr if sel(w) and st(z[m])]
            return (np.median(Y, 0), len(Y)) if Y else (None, 0)

        def err(y):
            if y is None:
                return np.nan
            s = (xm >= g.min()) & (xm <= g.max())
            return float(np.sqrt(np.mean((np.interp(xm[s], g, y) - ym[s]) ** 2)) / 2)
        r = dict(case=panel, **{c: old.loc[panel, c] for c in ("their TCL (full)", "their TC (full)")})
        r["TCL2_dressed"] = err(med("TCL2", lambda w: w == 0.01)[0]); r["TCL234_dressed"] = err(med("TCL234", lambda w: w == 0.01)[0])
        r["TCL2_RF"] = err(med("TCL2")[0]); r["TCL234_RF"] = err(med("TCL234")[0]); r["TEMPO"] = err(tem); rows.append(r)
    L = pd.DataFrame(rows).set_index("case"); pd.set_option("display.width", 220); print(L.round(4).to_string())
    L.to_csv(os.path.join(A.data, "lg_recomputed.csv"))
    print("\nmeans over %d cases:" % len(L)); print(L.mean().round(4).to_string())
    print("RF better than their TC: TCL2 %d / TCL234 %d of %d" % ((L.TCL2_RF < L["their TC (full)"]).sum(), (L.TCL234_RF < L["their TC (full)"]).sum(), len(L)))
    dr = L.dropna(subset=["TCL234_dressed"])
    print("dressed-frame cases with TCL234 stable: %d; TCL234 %.3f vs TCL2 %.3f vs their TC %.3f" % (len(dr), dr.TCL234_dressed.mean(), dr.TCL2_dressed.mean(), dr["their TC (full)"].mean()))
    print("TCL2 dressed mean over all: %.3f" % L.TCL2_dressed.mean())


if __name__ == "__main__":
    main()
