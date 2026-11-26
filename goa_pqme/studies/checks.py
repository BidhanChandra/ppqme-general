"""Trace and Hermiticity of the propagated reduced state rho~_S(t) (TCL2 and TCL234) on a sample of frames.

Reports  max_t |Tr rho - 1|,  max_t |rho_DA - conj(rho_AD)|  and the minimum population, over the frames that pass
the screen |dP| <= 1.6 (divergent frames are reported separately).

  python -m goa_pqme.studies.checks --nproc 4                          (3 frames x 8 cases, a few minutes)
  python -m goa_pqme.studies.checks --cases G0p2_OD0p28125_B2_y0p75 --all-frames
Output: <out>/trace_herm_check.csv  (default work/checks).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..config import ALPHAS, CONVERGED, OMEGAHS, WORK
from ..cases import get_case
from ..metrics import is_stable
from ..scan import run_jobs
from .tempo_conv import case_tmax

SAMPLE_CASES = ["G0p2_OD0_Binf_y0p75", "G0p2_OD0p28125_Binf_y0p75", "G1_OD0_Binf_y0p75", "G0p2_OD0_B5_y0p75",
                "G0p2_OD0p28125_B2_y0p75", "G0p2_OD0_B1_y0p75", "G0p2_OD0p28125_B0p5_y0p75", "G0p2_OD0_B10_y0p75"]
SAMPLE_FRAMES = [(0.5, 0.01), (2.0, 0.15), (8.0, 0.8)]       # strongly dressed, intermediate, lightly dressed


def one(job):
    from ..pipeline import run_frame
    cid, a, w, tmax = job
    res = run_frame(get_case(cid), a, w, CONVERGED.with_(tmax=tmax), keep_rho=("TCL2", "TCL234"))
    out = []
    for name in ("TCL2", "TCL234"):
        rs, dP = res.rho[name], res.dP[name]
        out.append(dict(case=cid, alpha=a, omega_h=w, method=name, B=res.B, stable=is_stable(dP),
                        trace_dev=float(np.nanmax(np.abs(rs[:, 0, 0] + rs[:, 1, 1] - 1))),
                        herm_dev=float(np.nanmax(np.abs(rs[:, 0, 1] - np.conj(rs[:, 1, 0])))),
                        min_pop=float(np.nanmin(np.minimum(rs[:, 0, 0].real, rs[:, 1, 1].real))),
                        max_abs_dP=float(np.nanmax(np.abs(dP)))))
    return out


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None); ap.add_argument("--nproc", type=int, default=2); ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--all-frames", action="store_true"); ap.add_argument("--out", default=os.path.join(WORK, "checks"))
    a = ap.parse_args(argv)
    ids = a.cases.split(",") if a.cases else SAMPLE_CASES
    frames = [(x, y) for x in ALPHAS for y in OMEGAHS] if a.all_frames else SAMPLE_FRAMES
    jobs = [(c, fa, fw, a.tmax or case_tmax(c)) for c in ids for fa, fw in frames]
    res = [r for rr in run_jobs(one, jobs, a.nproc, "frames") for r in rr]
    D = pd.DataFrame(res); os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "trace_herm_check.csv"); D.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    S = D[D.stable]
    print("frames checked:", len(D) // 2, "| passing the screen:", len(S) // 1)
    print(S.groupby("method")[["trace_dev", "herm_dev"]].agg(["max", "median"]).to_string(float_format=lambda x: "%.1e" % x))
    print("min population over stable frames:", S.groupby("method").min_pop.min().to_dict())
    if (~D.stable).any():
        print("divergent frames (not used): trace_dev median %.1e" % D[~D.stable].trace_dev.median())
    print("\nworst case per method:\n", S.sort_values("trace_dev").groupby("method").tail(1)[["case", "alpha", "omega_h", "method", "trace_dev", "herm_dev"]].to_string(index=False))
    print("written", path)


if __name__ == "__main__":
    main()
