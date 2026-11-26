"""Build the figures folder (inputs of GOA_figures.ipynb and of every other analysis) from a frame scan.

  python -m goa_pqme.analysis.build frames    --scan work/scan [--tempo-dir notebooks/csv] [--out goa_figs_v2]
  python -m goa_pqme.analysis.build coherence --scan work/scan [--coh-scan DIR] [--out goa_figs_v2]

frames:    (1) copy every TEMPO csv from --tempo-dir to OUT/csv (old *_frame_errors.csv are not copied), (2) per case write
           OUT/csv/<tag>_frame_errors.csv  (err = 0.5 RMSE(dP - dP_TEMPO) on the TEMPO grid, stable = finite & max|dP| <= 1.6),
           (3) write the per-frame curves OUT/curves/<case>_<alpha>_<omega_h>.npz, (4) print old-vs-new oracle / typical errors.
coherence: OUT/csv/<tag>_coherence_errors.csv: error = rms|rho_DA - rho_DA^TEMPO|, oracle / typical / RF (pointwise median of
           Re and Im) over the frames stable for the method; readouts 'B*rho~_DA' and 'B*rho~_DA + Q1'.
--scan may be a new goa_pqme scan or a legacy work/scan_new; --coh-scan defaults to --scan (new scans store the coherence
in the same files; for legacy data pass work/scan_coh).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np

from ..config import ALPHAS, OMEGAHS
from ..io import load_frame
from ..metrics import frame_table
from .common import FIGS_DIR, TEMPO_DIR, VARIANTS4, figure_keys, read_reference, stem_of, tempo_file


def _scan_file(scan, stem, al, w):
    return os.path.join(scan, "curves", "%s__a%g_w%g.npz" % (stem, al, w))


def build_frames(scan, tempo_dir, out):
    import pandas as pd
    os.makedirs(os.path.join(out, "csv"), exist_ok=True); os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    for f in glob.glob(os.path.join(tempo_dir, "*.csv")):
        if not f.endswith("_frame_errors.csv"):
            shutil.copy(f, os.path.join(out, "csv"))
    keys = sorted({os.path.basename(f).split("_tempo")[0] for f in glob.glob(os.path.join(tempo_dir, "*_tempo_T*_dt0.1_*.csv"))})
    if not keys:
        raise SystemExit("no <key>_tempo_T*_dt0.1_*.csv files in %s" % tempo_dir)
    print("%d cases found from TEMPO files" % len(keys))
    summ = []
    for key in keys:
        stem = stem_of(key); ref = read_reference(tempo_file(tempo_dir, key)); cur = {}; miss = 0
        for al in ALPHAS:
            for w in OMEGAHS:
                f = _scan_file(scan, stem, al, w)
                if not os.path.exists(f):
                    miss += 1; continue
                z = load_frame(f)
                cur[(float(al), float(w))] = dict(t=z["t"], r3=z["r3"], r4=z["r4"], **{v: z["dP"][v] for v in VARIANTS4})
                np.savez_compressed(os.path.join(out, "curves", "%s_%s_%s.npz" % (stem, float(al), float(w))),
                                    t=z["t"], **{v: z["dP"][v] for v in VARIANTS4})
        if not cur:
            print("NO CURVES for", key); continue
        new, *_ = frame_table(ref.t.values, ref.dP.values, cur)
        new.to_csv(os.path.join(out, "csv", key + "_frame_errors.csv"), index=False)
        oldf = os.path.join(tempo_dir, key + "_frame_errors.csv"); old = pd.read_csv(oldf) if os.path.exists(oldf) else None

        def ot(df, m):
            ok = df[df["stable_" + m]]
            return (ok["err_" + m].min(), ok["err_" + m].median(), len(ok)) if len(ok) else (np.nan, np.nan, 0)
        nan3 = (np.nan, np.nan, 0)
        o, n = (ot(old, "TCL234") if old is not None else nan3), ot(new, "TCL234")
        o2, n2 = (ot(old, "TCL2") if old is not None else nan3), ot(new, "TCL2")
        summ.append(dict(case=stem, frames_new=len(new), missing=miss, TCL234_oracle_old=o[0], TCL234_oracle_new=n[0], TCL234_typ_old=o[1],
                         TCL234_typ_new=n[1], stable234_old=o[2], stable234_new=n[2], TCL2_typ_old=o2[1], TCL2_typ_new=n2[1]))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(out, "old_vs_new_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.round(4).to_string(index=False)); print("\nwritten to", out)


def build_coherence(coh_scan, out):
    import pandas as pd
    csv_dir = os.path.join(out, "csv")
    for key in figure_keys(csv_dir):
        stem = stem_of(key); fe = pd.read_csv(os.path.join(csv_dir, key + "_frame_errors.csv")); ref = read_reference(tempo_file(csv_dir, key))
        if "Re_rho_DA" not in ref.columns:
            print("skip (no coherence in TEMPO file):", key); continue
        coh = {}
        for r in fe.itertuples():
            f = _scan_file(coh_scan, stem, r.alpha, r.omega_h)
            if os.path.exists(f):
                z = load_frame(f)
                if z["coherence"]:
                    coh[(r.alpha, r.omega_h)] = dict(t=z["t"], **z["coherence"])
        if not coh:
            print("no coherence curves for", key); continue
        t = coh[next(iter(coh))]["t"]; tc = t[t <= ref.t.iloc[-1] + 1e-9]
        rho = np.interp(tc, ref.t, ref.Re_rho_DA) + 1j * np.interp(tc, ref.t, ref.Im_rho_DA); refmax = np.abs(rho).max()
        rows = []
        for m in ("TCL2", "TCL234"):
            for ro, name in (("B", "B*rho~_DA"), ("Q1", "B*rho~_DA + Q1")):
                fr = [(al, w) for (al, w) in coh if bool(fe[(fe.alpha == al) & (fe.omega_h == w)]["stable_" + m].iloc[0])
                      and np.all(np.isfinite(coh[(al, w)][m + "_" + ro]))]
                if not fr:
                    continue
                Y = np.array([coh[k][m + "_" + ro][:len(tc)] for k in fr]); err = np.sqrt(np.mean(np.abs(Y - rho[None]) ** 2, axis=1))
                med = np.median(Y.real, 0) + 1j * np.median(Y.imag, 0); rf = np.sqrt(np.mean(np.abs(med - rho) ** 2))
                rows.append(dict(method=m, readout=name, frames=len(fr), oracle=err.min(), oracle_frame=fr[int(err.argmin())],
                                 typical=np.median(err), RF=rf, RF_percent_of_max=100 * rf / refmax))
        pd.DataFrame(rows).to_csv(os.path.join(csv_dir, key + "_coherence_errors.csv"), index=False)
        print("%-45s frames with coherence: %d" % (stem, len(coh)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["frames", "coherence"])
    ap.add_argument("--scan", required=True); ap.add_argument("--coh-scan", default=None)
    ap.add_argument("--tempo-dir", default=TEMPO_DIR); ap.add_argument("--out", default=FIGS_DIR)
    a = ap.parse_args(argv)
    if a.what == "frames":
        build_frames(a.scan, a.tempo_dir, a.out)
    else:
        build_coherence(a.coh_scan or a.scan, a.out)


if __name__ == "__main__":
    main()
