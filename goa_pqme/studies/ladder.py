"""Numerical-convergence ladder for the K3/K4 quadrature and the sparse-node interpolation.

For every case x frame the production setting and a ladder of finer settings (one knob at a time) are run and compared
with the finest setting 'ref':  relK_n = max over common nodes of ||K_n - K_n^ref|| / ||K_n^ref||,
maxdP = max_t |dP_TCL234 - dP_TCL234^ref|, and (if a TEMPO reference exists) the half-RMSE error of each setting.

  python -m goa_pqme.studies.ladder --list-settings
  python -m goa_pqme.studies.ladder --nproc 4                                  # all cases with TEMPO data, 6 default frames
  python -m goa_pqme.studies.ladder --cases G0p2_OD0_B1_y0p75 --frames 0.5:0.05,2:0.5
  python -m goa_pqme.studies.ladder --summary-only
Outputs (resumable): <out>/runs/*.npz, ladder_all.csv, ladder_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from ..cases import get_case, load_cases, load_reference, reference_file
from ..config import WORK, Numerics
from ..io import frame_K_total, load_frame, save_frame
from ..metrics import half_rmse
from ..scan import run_jobs

# name: (node_dt, nq, panel_h).  'prod' = paper production; 'ref' = finest, everything is compared to it.
SETTINGS = {
    "prod": (3.0, 4, 6.0),
    "node1.5": (1.5, 4, 6.0), "node1.0": (1.0, 4, 6.0),
    "nq6": (3.0, 6, 6.0),
    "h3": (3.0, 4, 3.0), "h2": (3.0, 4, 2.0),
    "mid": (1.5, 6, 3.0),
    "ref": (1.0, 6, 3.0),
}
DEFAULT_FRAMES = "0.5:0.05,0.5:0.29,1.5:0.095,2:0.5,4:0.15,8:0.02"


def run_path(outdir, cid, a, w, name, tmax):
    return os.path.join(outdir, "runs", "%s__a%g_w%g_%s_T%g.npz" % (cid, a, w, name, tmax))


def run_one(job):
    from ..pipeline import run_frame
    cid, a, w, name, tmax, outdir = job
    fn = run_path(outdir, cid, a, w, name, tmax)
    if os.path.exists(fn):
        return fn
    nd, nq, hp = SETTINGS[name]
    num = Numerics(node_dt=nd, nq=nq, panel_h=hp, dt=0.05, tmax=tmax)
    p = get_case(cid)
    save_frame(fn, run_frame(p, a, w, num), p, num)
    return fn


def tempo_err(cid, t, y):
    try:
        ref = load_reference(cid)
        return half_rmse(t, y, ref.t.values, ref.dP.values)
    except Exception:
        return float("nan")


def summarise(cases, frames, tmax, outdir):
    rows = []
    for cid in cases:
        for a, w in frames:
            f = lambda n: run_path(outdir, cid, a, w, n, tmax)
            if not os.path.exists(f("ref")):
                continue
            R = load_frame(f("ref"))
            eps_ref = tempo_err(cid, R["t"], R["dP"]["TCL234"])
            for n in SETTINGS:
                if n == "ref" or not os.path.exists(f(n)):
                    continue
                Z = load_frame(f(n))
                common = [t for t in Z["t_nodes"] if t > 0 and np.any(np.isclose(R["t_nodes"], t))]

                def rel(k):
                    e = []
                    KZ, KR = frame_K_total(Z, k), frame_K_total(R, k)
                    for t in common:
                        i = int(np.argmin(abs(Z["t_nodes"] - t)))
                        j = int(np.argmin(abs(R["t_nodes"] - t)))
                        den = np.linalg.norm(KR[j])
                        e.append(np.linalg.norm(KZ[i] - KR[j]) / den if den > 0 else 0.0)
                    return max(e) if e else float("nan")
                n_ = min(len(Z["t"]), len(R["t"]))
                rows.append(dict(case=cid, alpha=a, omega_h=w, setting=n, node_dt=SETTINGS[n][0], nq=SETTINGS[n][1], panel_h=SETTINGS[n][2],
                                 relK3_max=rel(3), relK4_max=rel(4),
                                 maxdP_TCL234_vs_ref=float(np.nanmax(np.abs(Z["dP"]["TCL234"][:n_] - R["dP"]["TCL234"][:n_]))),
                                 eps_TEMPO_this=tempo_err(cid, Z["t"], Z["dP"]["TCL234"]), eps_TEMPO_ref=eps_ref,
                                 runtime_s=float(Z.get("seconds", np.nan))))
    if not rows:
        print("nothing to summarise yet")
        return
    with open(os.path.join(outdir, "ladder_all.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summ = []
    for n in SETTINGS:
        rr = [r for r in rows if r["setting"] == n]
        if rr:
            summ.append(dict(setting=n, node_dt=SETTINGS[n][0], nq=SETTINGS[n][1], panel_h=SETTINGS[n][2], n_runs=len(rr),
                             worst_relK3=max(r["relK3_max"] for r in rr), worst_relK4=max(r["relK4_max"] for r in rr),
                             worst_maxdP=max(r["maxdP_TCL234_vs_ref"] for r in rr)))
    with open(os.path.join(outdir, "ladder_summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summ[0])); wr.writeheader(); wr.writerows(summ)
    print("\nWorst case over all cases/frames, vs the finest setting (ref):")
    print("%-8s %6s %3s %5s %5s %10s %10s %10s" % ("setting", "nodedt", "nq", "h", "runs", "relK3", "relK4", "maxdP"))
    for s in summ:
        print("%-8s %6g %3d %5g %5d %10.2e %10.2e %10.2e" % (s["setting"], s["node_dt"], s["nq"], s["panel_h"], s["n_runs"], s["worst_relK3"], s["worst_relK4"], s["worst_maxdP"]))
    print("\nwritten:", os.path.join(outdir, "ladder_all.csv"), "and ladder_summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all", help="comma list of case ids, or 'all' (cases with a TEMPO reference)")
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--settings", default=",".join(SETTINGS), help="subset; 'ref' and 'prod' are always included")
    ap.add_argument("--tmax", type=float, default=24.0)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(WORK, "ladder"))
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--list-settings", action="store_true")
    a = ap.parse_args(argv)
    if a.list_settings:
        for k, v in SETTINGS.items():
            print(k, "node_dt=%g nq=%d panel_h=%g" % v)
        return
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    cases = [c for c in load_cases() if reference_file(c)] if a.cases == "all" else a.cases.split(",")
    frames = [tuple(float(x) for x in s.split(":")) for s in a.frames.split(",")]
    names = list(dict.fromkeys(["prod", "ref"] + [s for s in a.settings.split(",") if s in SETTINGS]))
    if a.summary_only:
        return summarise(cases, frames, a.tmax, a.out)
    if a.timing:
        for n in names:
            t0 = time.time(); run_one((cases[0], frames[0][0], frames[0][1], n, a.tmax, a.out))
            print("%-8s %7.1f s" % (n, time.time() - t0), flush=True)
        return
    jobs = [(c, fa, fw, n, a.tmax, a.out) for c in cases for (fa, fw) in frames for n in names]
    cost = {n: SETTINGS[n][0] ** -1 * SETTINGS[n][1] ** 2 * (6.0 / SETTINGS[n][2]) for n in names}
    jobs.sort(key=lambda j: cost[j[3]])          # cheapest first, so partial results are usable early
    jobs = [j for j in jobs if not os.path.exists(run_path(a.out, j[0], j[1], j[2], j[3], j[4]))]
    print("%d cases x %d frames x %d settings; %d runs to do, nproc=%d" % (len(cases), len(frames), len(names), len(jobs), a.nproc), flush=True)
    run_jobs(run_one, jobs, a.nproc, "ladder")
    summarise(cases, frames, a.tmax, a.out)


if __name__ == "__main__":
    main()
