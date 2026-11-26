"""TEMPO time-step convergence for all cases (needs oqupy).  Resumable.

Production reference: dt = 0.1, epsrel 1e-8 (beta = inf) / 1e-9 (finite beta).  Halving dt accumulates truncation error,
so dt is varied together with a 10x tighter epsrel.  Two runs per case:
    A: dt = 0.1,  eps = prod/10      A vs production = effect of the truncation tolerance
    B: dt = 0.05, eps = prod/10      B vs A          = effect of the time step (same tolerance)
Errors are rms(dP difference)/2 on the coarser grid.

  python -m goa_pqme.studies.tempo_conv --nproc 4
  python -m goa_pqme.studies.tempo_conv --analyze [--prod-dir notebooks/csv]
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from ..cases import get_case, key_of, load_cases, reference_file
from ..config import TEMPO_DIR, WORK
from ..scan import run_jobs
from ..tempo import production_epsrel

OUT = os.path.join(WORK, "tempo_conv")


def fn(out, cid, dt, eps):
    return os.path.join(out, "%s__dt%g_eps%g.csv" % (cid, dt, eps))


def case_tmax(cid, tmax=None):
    """--tmax if given, else the last time of the case's TEMPO reference (capped at 24)."""
    from ..cases import case_tmax as _ct
    return float(tmax) if tmax else _ct(cid)


def job(a):
    from ..tempo import run_tempo
    cid, dt, eps, tmax, out = a
    f = fn(out, cid, dt, eps)
    if os.path.exists(f):
        return f
    t0 = time.time()
    df = run_tempo(get_case(cid), tmax=tmax, dt=dt, epsrel=eps)
    df.to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)
    print("%s dt=%g eps=%g  %.0f s" % (cid, dt, eps, time.time() - t0), flush=True)
    return f


def rms(t, d, ref):
    m = ref.t.values <= min(t.max(), ref.t.max()) + 1e-9
    return float(0.5 * np.sqrt(np.mean((np.interp(ref.t.values[m], t, d.dP.values) - ref.dP.values[m]) ** 2)))


def run(cases, nproc, tmax, out):
    cs = load_cases()
    ids = [c for c in cs if cs[c].Gamma > 0.05] if cases == "all" else cases.split(",")   # Gamma=0.01 is not a paper case
    ids.sort(key=lambda c: -min(cs[c].beta, 1e3))                                          # cheap (low T) first
    jobs = []
    for c in ids:
        e = production_epsrel(cs[c].beta) / 10; tm = case_tmax(c, tmax)
        jobs += [(c, 0.1, e, tm, out), (c, 0.05, e, tm, out)]
    os.makedirs(out, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(fn(out, j[0], j[1], j[2]))]
    print("%d runs, %d left; nproc=%d" % (len(jobs), len(todo), nproc), flush=True)
    run_jobs(job, todo, nproc, "TEMPO")


def analyze(prod_dir, out):
    import pandas as pd
    rows = []
    for c, p in load_cases().items():
        if p.Gamma <= 0.05:
            continue
        e = production_epsrel(p.beta) / 10; A, B = fn(out, c, 0.1, e), fn(out, c, 0.05, e)
        pf = sorted(f for f in glob.glob(os.path.join(prod_dir, key_of(c) + "_*tempo*")) if "dt0.1" in f)
        if not (os.path.exists(A) and os.path.exists(B)):
            rows.append(dict(case=c, status="missing runs")); continue
        dA, dB = pd.read_csv(A), pd.read_csv(B)
        for d in (dA, dB):
            if "t" not in d:
                d["t"] = d["t_au"]
        r = dict(case=c, beta=p.beta, dt_effect_B_vs_A=rms(dB.t.values, dB, dA))
        if pf:
            prod = pd.read_csv(pf[0])
            if "t" not in prod:
                prod["t"] = prod["t_au"]
            r.update(eps_truncation_A_vs_prod=rms(dA.t.values, dA, prod), total_B_vs_prod=rms(dB.t.values, dB, prod))
        else:
            r["status"] = "production csv not found (dt effect only)"
        rows.append(r)
    T = pd.DataFrame(rows); pd.set_option("display.width", 200); print(T.to_string(index=False, float_format=lambda x: "%.2e" % x))
    if "dt_effect_B_vs_A" in T:
        print("\nmax time-step effect over %d cases: %.2e" % (T.dt_effect_B_vs_A.notna().sum(), T.dt_effect_B_vs_A.max()))
    if "total_B_vs_prod" in T and T.total_B_vs_prod.notna().any():
        print("max over %d cases with production file: truncation %.2e | total %.2e" % (T.total_B_vs_prod.notna().sum(), T.eps_truncation_A_vs_prod.max(), T.total_B_vs_prod.max()))
    T.to_csv(os.path.join(out, "tempo_convergence_summary.csv"), index=False); print("written", os.path.join(out, "tempo_convergence_summary.csv"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all"); ap.add_argument("--nproc", type=int, default=2); ap.add_argument("--tmax", default=None)
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--prod-dir", default=TEMPO_DIR); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    analyze(a.prod_dir, a.out) if a.analyze else run(a.cases, a.nproc, a.tmax, a.out)


if __name__ == "__main__":
    main()
