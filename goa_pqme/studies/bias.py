"""Bias scan: does the failure of TCL234 switch on near zero bias omega_DA - E_r?
Fixed Gamma = 0.2, y0 = 0.75 (E_r = 0.28125); beta and omega_DA scanned; three frames per point:
    lab = (alpha 2, omega_h 1000) [f ~ 1e-5, B = 1],   mid = (3, 0.5),   dressed = (3, 0.05).

  python -m goa_pqme.studies.bias tempo   --nproc 4 --betas 5     TEMPO references  (dt 0.1, eps 1e-9 / 1e-8 at beta=inf)
  python -m goa_pqme.studies.bias tcl     --nproc 8 --betas 5     channel-resolved K2..K4 + all propagation variants
  python -m goa_pqme.studies.bias analyze                         errors, ratios, table and figure
  python -m goa_pqme.studies.bias lab --tempo-dir DIR --betas inf,10,5,2 --od 0.28125     lab-frame control (TCL2..TCL234 vs TEMPO)
Outputs in <out> (default work/bias_scan): tempo/*.csv, tcl/<key>__<frame>.npz, analysis/bias_scan.csv, fig_bias_scan.png.
Resumable; finished files are skipped.  'analyze' also reads files written by the legacy scan/bias_scan.py.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import numpy as np

from ..config import WORK, ModelParams, Numerics
from ..io import load_frame, save_frame
from ..metrics import half_rmse, is_stable
from ..scan import run_jobs

ER = 2 * 0.5 ** 2 * 0.75 ** 2          # 0.28125
GAMMA, Y0 = 0.2, 0.75
ODS = [0.0, 0.06, 0.12, 0.18, 0.22, 0.25, 0.27, 0.28125, 0.30, 0.34, 0.42]
FRAMES = {"lab": (2.0, 1000.0), "mid": (3.0, 0.5), "dressed": (3.0, 0.05)}
NUMERICS = Numerics(node_dt=1.5, nq=4, panel_h=3.0, dt=0.05, tmax=24.0)
GROUP_VARIANTS = ("TCL2", "K3", "K4", "TCL234", "hop", "lin", "mix")


def _tg(x):
    return ("%g" % x).replace(".", "p")


def key(beta, od):
    return "G0p2_OD%s_B%s_y0p75" % (_tg(od), "inf" if math.isinf(beta) else _tg(beta))


def params(beta, od) -> ModelParams:
    return ModelParams(Gamma=GAMMA, omega_DA=od, beta=beta, y0=Y0)


def _tempo_job(args):
    from ..tempo import production_epsrel, run_tempo
    beta, od, tmax, out = args
    f = os.path.join(out, "tempo", key(beta, od) + ".csv")
    if os.path.exists(f):
        return f
    t0 = time.time()
    df = run_tempo(params(beta, od), tmax=tmax, dt=0.1, epsrel=production_epsrel(beta))
    df.to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)
    print("TEMPO %s  %.0f s" % (key(beta, od), time.time() - t0), flush=True)
    return f


def _tcl_job(args):
    from ..pipeline import run_frame
    beta, od, fr, num, out = args
    f = os.path.join(out, "tcl", "%s__%s.npz" % (key(beta, od), fr))
    if os.path.exists(f):
        return f
    al, w = FRAMES[fr]
    p = params(beta, od)
    res = run_frame(p, al, w, num, resolve=True)
    save_frame(f, res, p, num, extra=dict(frame=fr))
    print("TCL %s %s  %.0f s" % (key(beta, od), fr, res.seconds), flush=True)
    return f


def _frame_scalars(z):
    """beta, omega_DA, frame label, B, gap, lambda_p, r4, tmax from a loaded (new or legacy) bias file."""
    beta, od = float(z["params"]["beta"]), float(z["params"]["omega_DA"])
    fr = str(z["extra"].get("frame", z.get("frame")))
    E = z["E"]
    tmax = float(z["numerics"].get("tmax", z["t"][-1]))
    return beta, od, fr, float(z["B"]), float(E[1] - E[0]), float(z["lambda_p"]), float(z["r4"]), tmax


def analyze(out):
    import pandas as pd
    ref, rows = {}, []
    for fnm in sorted(glob.glob(os.path.join(out, "tcl", "*.npz"))):
        z = load_frame(fnm)
        beta, od, fr, B, gap, lp, r4, tmax = _frame_scalars(z)
        k = key(beta, od)
        tf = os.path.join(out, "tempo", k + ".csv")
        if not os.path.exists(tf):
            continue
        if k not in ref:
            d = pd.read_csv(tf)
            ref[k] = (d["t"].values if "t" in d else d.iloc[:, 0].values, d["dP"].values)
        tr, yr = ref[k]
        sel = tr <= tmax + 1e-9
        r = dict(beta=beta, bias=od - ER, omega_DA=od, frame=fr, B=B, gap=gap, eps_t=od - ER + lp, r4=r4)
        for v in GROUP_VARIANTS:
            y = z["dP"][v]
            r["err_" + v] = half_rmse(z["t"], y, tr[sel], yr[sel]) if is_stable(y) else np.inf
        rows.append(r)
    if not rows:
        print("no TCL files with matching TEMPO references in", out); return
    F = pd.DataFrame(rows).sort_values(["beta", "frame", "bias"])
    F["ratio"] = F.err_TCL234 / F.err_TCL2; F["d_K4"] = F.err_K4 - F.err_TCL2
    for v in ("hop", "lin", "mix"):
        F["d_" + v] = F["err_" + v] - F.err_TCL2
    os.makedirs(os.path.join(out, "analysis"), exist_ok=True)
    F.round(6).to_csv(os.path.join(out, "analysis", "bias_scan.csv"), index=False)
    pd.set_option("display.width", 220)
    for (b, fr), d in F.groupby(["beta", "frame"]):
        print("\n=== beta=%g  frame=%s  (B=%.3f) ===" % (b, fr, d.B.iloc[0]))
        print(d[["bias", "eps_t", "gap", "r4", "err_TCL2", "err_K4", "err_TCL234", "ratio", "d_K4", "d_hop", "d_lin", "d_mix"]].round(4).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        bs = sorted(F.beta.unique()); fig, ax = plt.subplots(len(bs), 2, figsize=(11, 3.8 * len(bs)), squeeze=False)
        for i, b in enumerate(bs):
            for fr, c in (("lab", "#C9811A"), ("mid", "#6B5B95"), ("dressed", "#176B77")):
                d = F[(F.beta == b) & (F.frame == fr)]
                ax[i, 0].plot(d.bias, d.err_TCL2, "o--", c=c, label=fr + " TCL2"); ax[i, 0].plot(d.bias, d.err_TCL234, "s-", c=c, label=fr + " TCL234")
                ax[i, 1].plot(d.bias, d.ratio, "o-", c=c, label=fr)
            ax[i, 0].set_ylabel("error vs TEMPO"); ax[i, 0].set_title("beta=%g" % b); ax[i, 1].axhline(1, c="k", lw=.8); ax[i, 1].set_yscale("log"); ax[i, 1].set_ylabel("TCL234 / TCL2")
            for j in range(2):
                ax[i, j].set_xlabel("bias  omega_DA - E_r"); ax[i, j].axvline(0, c="grey", lw=.6)
            ax[i, 0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(out, "analysis", "fig_bias_scan.png"), dpi=150); print("figure written")
    except Exception as e:
        print("figure skipped:", e)


def lab_control(betas, od, tempo_dir, out, num: Numerics, nproc: int, tag: str = "lab"):
    """Explicit laboratory frame (alpha 2, omega_h 1000): TCL2, K3, K4, TCL234 against TEMPO, and K3/K2, K4/K2."""
    import pandas as pd
    from ..pipeline import run_frame
    os.makedirs(out, exist_ok=True)
    al, w = FRAMES["lab"]
    jobs = [(b, od, num, out, tag, al, w) for b in betas]
    files = run_jobs(_lab_job, jobs, nproc, "lab frames")
    rows = []
    for b, f in zip(betas, files):
        z = load_frame(f)
        btag = "inf" if math.isinf(b) else "%g" % b
        cand = glob.glob(os.path.join(tempo_dir, "G0.2_OD%g_B%s_y0.75_*tempo*.csv" % (od, btag)))
        if not cand:
            print("no TEMPO file for", btag); continue
        d = pd.read_csv(cand[0]); tr, yr = d["t"].values, d["dP"].values; sel = tr <= num.tmax + 1e-9
        from ..tcl_higher import rms_norm
        from ..io import frame_K_total
        k2, k3, k4 = (rms_norm(frame_K_total(z, n)) for n in (2, 3, 4))
        r = dict(beta=b, B=float(z["B"]), K3_over_K2=k3 / k2, K4_over_K2=k4 / k2)
        for v in ("TCL2", "K3", "K4", "TCL234"):
            r["err_" + v] = float(np.sqrt(np.mean((np.interp(tr[sel], z["t"], z["dP"][v]) - yr[sel]) ** 2)) / 2)
        rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.round(5).to_string(index=False))
    R.round(6).to_csv(os.path.join(out, tag + "_control.csv"), index=False)


def _lab_job(args):
    from ..pipeline import run_frame
    b, od, num, out, tag, al, w = args
    f = os.path.join(out, "%s_%s_OD%g.npz" % (tag, "inf" if math.isinf(b) else "%g" % b, od))
    if not os.path.exists(f):
        p = params(b, od)
        save_frame(f, run_frame(p, al, w, num), p, num, extra=dict(frame=tag))
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tempo", "tcl", "analyze", "lab"])
    ap.add_argument("--nproc", type=int, default=2); ap.add_argument("--betas", default="5")
    ap.add_argument("--omega-das", default=None); ap.add_argument("--tmax", type=float, default=24.0)
    ap.add_argument("--out", default=os.path.join(WORK, "bias_scan"))
    ap.add_argument("--tempo-dir", default=None, help="lab mode: folder with G0.2_OD.._B.._y0.75_*tempo*.csv")
    ap.add_argument("--od", type=float, default=0.28125, help="lab mode: omega_DA")
    ap.add_argument("--tag", default="lab")
    a = ap.parse_args(argv)
    betas = [math.inf if x == "inf" else float(x) for x in a.betas.split(",")]
    ods = [float(x) for x in a.omega_das.split(",")] if a.omega_das else ODS
    num = NUMERICS.with_(tmax=a.tmax)
    if a.mode == "lab":
        from ..config import TEMPO_DIR
        return lab_control(betas, a.od, a.tempo_dir or TEMPO_DIR, a.out if a.out != os.path.join(WORK, "bias_scan") else os.path.join(WORK, "lab_control"), num, a.nproc, a.tag)
    for d in ("tempo", "tcl", "analysis"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    if a.mode == "analyze":
        return analyze(a.out)
    if a.mode == "tempo":
        jobs = [(b, od, a.tmax, a.out) for b in betas for od in ods if not os.path.exists(os.path.join(a.out, "tempo", key(b, od) + ".csv"))]
        print("%d TEMPO runs left; nproc=%d" % (len(jobs), a.nproc), flush=True)
        run_jobs(_tempo_job, jobs, a.nproc, "TEMPO")
    else:
        jobs = [(b, od, fr, num, a.out) for b in betas for od in ods for fr in FRAMES
                if not os.path.exists(os.path.join(a.out, "tcl", "%s__%s.npz" % (key(b, od), fr)))]
        print("%d TCL frames left; nproc=%d" % (len(jobs), a.nproc), flush=True)
        run_jobs(_tcl_job, jobs, a.nproc, "TCL")


if __name__ == "__main__":
    main()
