"""Resumable frame scans.

    scan_frames(case_ids, numerics=CONVERGED, nproc=4)   ->  <out>/curves/<case>__a<alpha>_w<omega_h>.npz + <out>/summary.csv
    run_grid(params, alphas, omega_hs, numerics, ...)    ->  {(alpha, omega_h): curves}   (any ModelParams, cached; notebook use)

Finished frames are skipped unless overwrite=True; failures go to <out>/failed.log.  summary.csv always covers
EVERY frame file in <out>/curves (so re-running one case never truncates it).
Set OMP_NUM_THREADS=1 when running with several processes (the scripts in scripts/ do this).
"""
from __future__ import annotations

import csv
import os
import time
import traceback

import numpy as np

from .cases import case_id, case_tmax, get_case, load_reference, tag
from .config import ALPHAS, CONVERGED, OMEGAHS, WORK, ModelParams, Numerics
from .io import frame_filename, iter_frames, load_frame, save_frame
from .metrics import half_rmse, is_stable


def run_jobs(fn, jobs, nproc: int = 1, label: str = "jobs"):
    """Map fn over jobs (process pool if nproc > 1) with a progress line."""
    t0, out = time.time(), []
    if not jobs:
        return out
    if nproc > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(nproc) as ex:
            for k, r in enumerate(ex.map(fn, jobs, chunksize=1)):
                out.append(r)
                print("\r%s %d/%d  %.1f min" % (label, k + 1, len(jobs), (time.time() - t0) / 60), end="", flush=True)
    else:
        for k, j in enumerate(jobs):
            out.append(fn(j))
            print("\r%s %d/%d  %.1f min" % (label, k + 1, len(jobs), (time.time() - t0) / 60), end="", flush=True)
    print()
    return out


def _frame_job(job):
    from .pipeline import run_frame
    cid, a, w, numerics, out, coherence, resolve, overwrite = job
    path = frame_filename(out, cid, a, w)
    if os.path.exists(path) and not overwrite:
        return path
    try:
        p = get_case(cid)
        res = run_frame(p, a, w, numerics, coherence=coherence, resolve=resolve)
        save_frame(path, res, p, numerics)
    except Exception:
        with open(os.path.join(out, "failed.log"), "a") as fh:
            fh.write("%s %g %g\n%s\n" % (cid, a, w, traceback.format_exc()))
    return path


def stable_frames_from_summary(summary_csv: str, cid: str, methods=("TCL2", "TCL234")) -> set:
    """Frames of a case that are stable for any of `methods` in a summary.csv (legacy or new layout)."""
    S = set()
    with open(summary_csv) as fh:
        for r in csv.DictReader(fh):
            if r["case"] != cid:
                continue
            for m in methods:
                if "stable_" + m in r:
                    ok = r["stable_" + m] == "True"
                else:      # legacy scan_frames summary
                    ok = r["finite_" + m] == "True" and float(r["maxabs_" + m]) <= 1.6
                if ok:
                    S.add((float(r["alpha"]), float(r["omega_h"])))
                    break
    return S


def scan_frames(case_ids, alphas=ALPHAS, omegahs=OMEGAHS, numerics: Numerics = CONVERGED, out: str | None = None,
                nproc: int = 1, coherence: bool = True, resolve: bool = False, only: dict | None = None,
                overwrite: bool = False, per_case_tmax: bool = True) -> str:
    """only: optional {case_id: set of (alpha, omega_h)} restricting the frames.
    per_case_tmax: horizon = min(numerics.tmax, last time of the case's TEMPO reference) (e.g. 6 for Gamma = 1)."""
    out = out or os.path.join(WORK, "scan")
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    num_of = {c: (numerics.with_(tmax=min(numerics.tmax, case_tmax(c, numerics.tmax))) if per_case_tmax else numerics) for c in case_ids}
    jobs = [(c, float(a), float(w), num_of[c], out, coherence, resolve, overwrite)
            for c in case_ids for a in alphas for w in omegahs
            if only is None or (float(a), float(w)) in only.get(c, set())]
    todo = jobs if overwrite else [j for j in jobs if not os.path.exists(frame_filename(out, j[0], j[1], j[2]))]
    print("%d frames, %d to do  (%s, coherence=%s, resolve=%s)" % (len(jobs), len(todo), numerics, coherence, resolve), flush=True)
    run_jobs(_frame_job, todo, nproc, "frames")
    summarise(out)
    return out


def summarise(out: str) -> str | None:
    """<out>/summary.csv: one row per frame file: r3, r4, B, stability, max|dP| and errors vs TEMPO (if available)."""
    rows, refs = [], {}
    for path, cid, a, w in iter_frames(out):
        z = load_frame(path)
        if cid not in refs:
            try:
                refs[cid] = load_reference(cid)
            except FileNotFoundError:
                refs[cid] = None
        ref = refs[cid]
        r = dict(case=cid, alpha=a, omega_h=w, r3=float(z.get("r3", np.nan)), r4=float(z.get("r4", np.nan)),
                 B=float(z.get("B", np.nan)), tmax=float(z["t"][-1]), seconds=float(z.get("seconds", np.nan)))
        for m, y in z["dP"].items():
            r["stable_" + m] = is_stable(y)
            r["maxabs_" + m] = float(np.nanmax(np.abs(y))) if np.all(np.isfinite(y)) else float("inf")
            if ref is not None:
                r["err_" + m] = half_rmse(z["t"], y, ref.t.values, ref.dP.values)
        rows.append(r)
    if not rows:
        return None
    path = os.path.join(out, "summary.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("summary:", path, "(%d frames)" % len(rows))
    return path


# ---------------------------------------------------------------------------------------------
# Arbitrary parameters (notebook): cached grid of frames
# ---------------------------------------------------------------------------------------------
def _grid_name(p: ModelParams, a, w, n: Numerics) -> str:
    return "%s__a%g_w%g_T%g_nd%g_nq%d_h%g.npz" % (tag(p), a, w, n.tmax, n.node_dt, n.nq, n.panel_h)


def _grid_job(job):
    from .pipeline import run_frame
    p, a, w, n, cache = job
    fn = os.path.join(cache, "curves", _grid_name(p, a, w, n))
    if not os.path.exists(fn):
        save_frame(fn, run_frame(p, a, w, n), p, n, keep_K=True)
    z = load_frame(fn)
    return a, w, dict(t=z["t"], r3=z["r3"], r4=z["r4"], seconds=z.get("seconds", 0.0), **z["dP"])


def run_grid(p: ModelParams, alphas, omegahs, numerics: Numerics = CONVERGED, nproc: int = 1, cache: str = "frames_cache") -> dict:
    """{(alpha, omega_h): {'t', 'TCL2', 'K3', 'K4', 'TCL234', 'r3', 'r4'}}; frames are cached in <cache>/curves."""
    os.makedirs(os.path.join(cache, "curves"), exist_ok=True)
    jobs = [(p, float(a), float(w), numerics, cache) for a in alphas for w in omegahs]
    return {(a, w): o for a, w, o in run_jobs(_grid_job, jobs, nproc, "frames")}


def print_model(p: ModelParams, alpha: float | None = None, omega_h: float | None = None):
    """Print the lab Hamiltonian, bath summary and (optionally) one transformed frame; returns (omega, R, E_r)."""
    from .bath import normal_modes
    from .frame import lab_hamiltonian, make_frame
    m = normal_modes(p)
    print("GOA parameters:", p, "  case id:", case_id(p))
    print("\nLab-frame Hamiltonian  H = H_S + H_B + |A><A| (x) X,   H_S (basis D, A):")
    print(lab_hamiltonian(p).real)
    print("  H_B = sum_q (p_q^2 + w_q^2 q_q^2)/2,   X = - sum_q w_q^2 R_q q_q")
    print("  reorganisation energy E_r = 1/2 sum w_q^2 R_q^2 = %.6f   (2 Omega^2 y0^2 = %.6f)" % (m.Er, p.Er))
    print("  N = %d normal modes, w in [%.4f, %.4f]" % (len(m.omega), m.omega.min(), m.omega.max()))
    print("  initial state: |D><D| (x) thermal bath on the donor surface, beta = %g (inf = vacuum)" % p.beta)
    if alpha is not None:
        fr = make_frame(p, alpha, omega_h)
        print("\nFrame (alpha=%g, omega_h=%g):  B = %.6f,  lambda_p = %.6f" % (alpha, omega_h, fr.B, fr.lambda_p))
        print("Transformed system Hamiltonian  H~_S = [[w_DA, Gamma B],[Gamma B, E_r - lambda_p]]:")
        print(fr.H_tilde.real)
        print("  eigenvalues:", fr.E)
    return m.omega, m.R, m.Er
