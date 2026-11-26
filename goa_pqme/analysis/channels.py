"""Channel-resolved analysis: which vertex class of K3/K4 drives the TCL234 error, and how it depends on the parameters.

  python -m goa_pqme.analysis.channels --scan work/scan_channels [--tempo-dir notebooks/csv] [--out work/channels_analysis]
Input: a scan run with --resolve (new format) or the legacy work/scan_channels.  Per frame:
  err_<variant>   0.5 RMSE of dP against TEMPO for TCL2 and TCL2 + channel classes (nX = number of linear vertices;
                  hop = all-hopping classes, lin = all-linear classes, mix = the cross classes);  d_<v> = err_<v> - err_TCL2
  rho<n>_x<c>     rms norm of the class-c part of K_n relative to rms ||K2||
  kap<n>_x<c>     population relaxation contribution -(K00 + K33) of the class, mean over nodes t >= 12
  parameters      B, lam_p, gap, Er, Er_res = 0.5 sum w^2 R^2 (1-f)^2, sx = sqrt(C_X(0)), eps_h = Gamma sqrt(1-B^2)/gap,
                  g_lin = Er_res/gap, s_lin = sx/gap, mixing = 2 Gamma B/gap, beta*gap
Outputs: frames.csv, tables.txt and fig_channels.png in --out.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ..config import ModelParams, WORK
from ..frame import make_frame
from ..io import iter_frames, load_frame
from .common import TEMPO_DIR, key_of, read_reference

GROUPS = ["K3", "K4", "TCL234", "hop", "lin", "mix"]
CLS = ["K3_x%d" % c for c in range(4)] + ["K4_x%d" % c for c in range(5)]
VARS = ["TCL2"] + GROUPS + CLS


def nrm(K):
    return np.sqrt(np.mean(np.linalg.norm(K[1:], axis=(1, 2)) ** 2))


def main(argv=None):
    import pandas as pd
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default=os.path.join(WORK, "scan_channels")); ap.add_argument("--tempo-dir", default=TEMPO_DIR)
    ap.add_argument("--tempo-glob", default=None, help="override; placeholders {dir}, {cid}, {key}")
    ap.add_argument("--out", default=os.path.join(WORK, "channels_analysis")); ap.add_argument("--stable-max", type=float, default=1.6)
    ap.add_argument("--min-gamma", type=float, default=0.05)
    A = ap.parse_args(argv)
    os.makedirs(A.out, exist_ok=True)
    _ref, _warned = {}, set()

    def ref(cid):
        if cid not in _ref:
            key = key_of(cid)
            pats = [A.tempo_glob] if A.tempo_glob else ["{dir}/{cid}.csv", "{dir}/{cid}_*.csv", "{dir}/{key}*tempo*dt0.1*.csv", "{dir}/{key}*dt0.1*.csv", "{dir}/{key}*.csv"]
            fs = []
            for pt in pats:
                fs = sorted(glob.glob(pt.format(dir=A.tempo_dir, cid=cid, key=key)))
                if fs:
                    break
            if not fs:
                raise FileNotFoundError("no TEMPO file for %s" % cid)
            print("TEMPO for %-28s <- %s" % (cid, os.path.basename(fs[0])))
            d = read_reference(fs[0])
            y = d["dP"].values if "dP" in d.columns else (d["P_D"] - d["P_A"]).values
            _ref[cid] = (d["t"].values, y)
        return _ref[cid]

    rows = []
    for fn, cid, a, w in iter_frames(A.scan):
        z = load_frame(fn)
        if not z.get("resolved"):
            continue
        prm = z["params"]
        if float(prm["Gamma"]) < A.min_gamma:
            continue
        try:
            tr, yr = ref(cid)
        except FileNotFoundError as e:
            if cid not in _warned:
                _warned.add(cid); print("SKIPPED", e)
            continue
        p = ModelParams(**{k: (int(v) if k == "N" else float(v)) for k, v in prm.items()})
        if "Er_res" in z:
            Er, Er_res, cx0 = float(z["Er"]), float(z["Er_res"]), float(z["cx0"])
        else:
            frm = make_frame(p, a, w); om, R = frm.omega, frm.modes.R
            Er, Er_res, cx0 = p.Er, float(0.5 * np.sum(om ** 2 * R ** 2 * (1 - frm.f) ** 2)), float(frm.linear([0.0])[0].real)
        tmax = float(z["numerics"].get("tmax", z["t"][-1])); sel = tr <= tmax + 1e-9; tt, rd = tr[sel], yr[sel]
        r = dict(case=cid, alpha=float(a), omega_h=float(w), Gamma=p.Gamma, omega_DA=p.omega_DA, beta=p.beta, B=float(z["B"]), lam_p=float(z["lambda_p"]),
                 Er=Er, gap=float(z["E"][1] - z["E"][0]), Er_res=Er_res, sx=float(np.sqrt(cx0)))
        r["regime"] = "activationless" if abs(r["omega_DA"] - r["Er"]) < 1e-9 else ("normal" if r["omega_DA"] < r["Er"] else "inverted")
        r["bclass"] = "beta>=2" if r["beta"] >= 2 else "beta<=1"; r["cell"] = (r["regime"] == "activationless") and r["beta"] >= 2
        r["eps_h"] = r["Gamma"] * np.sqrt(max(1 - r["B"] ** 2, 0)) / r["gap"]; r["g_lin"] = r["Er_res"] / r["gap"]; r["s_lin"] = r["sx"] / r["gap"]
        r["mixing"] = 2 * r["Gamma"] * r["B"] / r["gap"]; r["beta_gap"] = r["beta"] * r["gap"] if np.isfinite(r["beta"]) else 1e6; r["lamp_Er"] = r["lam_p"] / r["Er"]
        n2 = nrm(z["K"][2].sum(0)); sel_n = z["t_nodes"] >= 12
        for n in (2, 3, 4):
            for c in range(n + 1):
                K = z["K"][n][c]; r["rho%d_x%d" % (n, c)] = nrm(K) / n2; r["kap%d_x%d" % (n, c)] = float(-(K[sel_n, 0, 0] + K[sel_n, 3, 3]).real.mean())
        for v in VARS:
            y = z["dP"][v]; fin = bool(np.all(np.isfinite(y))); yi = np.interp(tt, z["t"], y)
            r["stable_" + v] = fin and bool(np.nanmax(np.abs(y)) <= A.stable_max); r["err_" + v] = float(np.sqrt(np.mean((yi - rd) ** 2)) / 2) if fin else np.inf
        rows.append(r)
    F = pd.DataFrame(rows)
    for v in VARS[1:]:
        F["d_" + v] = np.where(F.stable_TCL2 & F["stable_" + v], F["err_" + v] - F.err_TCL2, np.nan)
    F["ratio"] = np.where(F.stable_TCL2 & F.stable_TCL234, F.err_TCL234 / F.err_TCL2, np.nan)
    F.round(6).to_csv(os.path.join(A.out, "frames.csv"), index=False)
    out = []

    def P(*s):
        print(*s); out.append(" ".join(str(x) for x in s))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    P("frames: %d  cases: %d " % (len(F), F.case.nunique()))
    P("\n=== 1. Effect of adding each channel group to TCL2, by regime and temperature (median d = err(TCL2+group) - err(TCL2); fraction of frames with d>0) ===")
    G = F.groupby(["regime", "bclass"])
    t1 = G.apply(lambda d: pd.Series({**{"n": len(d), "ratio_med": d.ratio.median()}, **{("d_" + v): d["d_" + v].median() for v in GROUPS},
                                      **{("worse_" + v): (d["d_" + v] > 0).sum() / max(d["d_" + v].notna().sum(), 1) for v in ("K4", "hop", "lin", "mix")}}))
    P(t1.round(4).to_string())
    P("\n=== 2. Individual classes (median d), activationless beta>=2 cell against the rest ===")
    t2 = F.groupby("cell").apply(lambda d: pd.Series({("d_" + v): d["d_" + v].median() for v in CLS})); P(t2.round(4).T.to_string())
    P("\n=== 3. Share of the K4 and K2 norm by class, against dressing (bins of B) ===")
    F["Bbin"] = pd.cut(F.B, [0, 0.3, 0.6, 0.9, 0.99, 1.0001])
    cols = ["rho2_x%d" % c for c in range(3)] + ["rho4_x%d" % c for c in range(5)] + ["kap2_x0", "kap2_x1", "kap2_x2", "kap4_x0", "kap4_x2", "kap4_x4"]
    P(F.groupby(["cell", "Bbin"], observed=True)[["ratio"] + cols].median().round(4).to_string())
    P("\n=== 4. Spearman correlation of the class effects with the physical parameters (frames stable for TCL2 and the variant) ===")
    PAR = ["B", "lamp_Er", "gap", "g_lin", "s_lin", "eps_h", "mixing", "beta_gap", "Gamma", "omega_DA"]
    for lab, sub in (("cell", F[F.cell]), ("all frames", F)):
        rr = {}
        for v in ("TCL234", "K4", "hop", "lin", "mix"):
            d = sub[["d_" + v] + PAR].dropna(); rr[v] = {p_: (spearmanr(d["d_" + v], d[p_])[0] if len(d) > 10 and d[p_].nunique() > 1 else np.nan) for p_ in PAR}
        P("\n--", lab, "(n=%d)" % len(sub)); P(pd.DataFrame(rr).round(2).to_string())
    P("\n=== 5. In the cell, frames where TCL234 is worse than TCL2: which group has the largest positive d ===")
    w_ = F[F.cell & (F.ratio > 1)]
    if len(w_):
        P("n=%d; largest-d group:" % len(w_), w_[["d_hop", "d_lin", "d_mix"]].idxmax(axis=1).value_counts().to_dict(), "| median d: ", w_[["d_hop", "d_lin", "d_mix", "d_TCL234"]].median().round(4).to_dict())
    open(os.path.join(A.out, "tables.txt"), "w").write("\n".join(out))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8)); C = {"hop": "#176B77", "lin": "#C9811A", "mix": "#6B5B95"}
        for g in ("hop", "lin", "mix"):
            s = F[F.cell]; ax[0].scatter(s.B, s["d_" + g], s=14, c=C[g], label=g, alpha=.7)
        ax[0].axhline(0, c="k", lw=.8); ax[0].set_xlabel("B (1 = undressed)"); ax[0].set_ylabel("err(TCL2 + group) - err(TCL2)"); ax[0].set_title("activationless beta>=2: effect of each channel group"); ax[0].legend()
        for g, lab in (("rho4_x0", "hopping only"), ("rho4_x2", "2 linear"), ("rho4_x4", "linear only")):
            s = F[F.cell]; ax[1].scatter(s.B, s[g], s=14, label=lab, alpha=.7)
        ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].set_xlabel("B"); ax[1].set_ylabel("||K4 class|| / ||K2||"); ax[1].set_title("size of K4 classes"); ax[1].legend()
        s = F.dropna(subset=["d_lin"]); sc = ax[2].scatter(s.g_lin, s.d_lin, s=14, c=np.log10(np.where(np.isfinite(s.beta), s.beta, 100)), cmap="viridis")
        ax[2].set_xscale("log"); ax[2].axhline(0, c="k", lw=.8); ax[2].set_xlabel("g_lin = E_r,res / gap"); ax[2].set_ylabel("d_lin"); ax[2].set_title("linear classes vs residual linear coupling (colour: log10 beta)"); plt.colorbar(sc, ax=ax[2])
        plt.tight_layout(); plt.savefig(os.path.join(A.out, "fig_channels.png"), dpi=150); print("figure written")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
