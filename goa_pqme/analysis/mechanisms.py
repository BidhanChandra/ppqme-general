"""Tests of the proposed failure mechanisms of TCL2 / TCL234.

  python -m goa_pqme.analysis.mechanisms [--data goa_figs_v2] [--generators 12]       -> <data>/mechanism_*.csv

A. r4 as the control parameter (frame-error CSVs only)
   A1  TCL234 / K4 stability against r4;   A2  P(TCL234 worse than TCL2 | both stable) against r4, and the
       (activationless, beta >= 2) cell effect at fixed r4.
B. spectral proxy (modes only): Spearman of the error ratio with the residual bath weight at the transformed gap
       S_res(Delta~) = sum_q (w_q R_q)^2 (1 - f_q)^2 coth(beta w_q/2) N(w_q; Delta~, sigma).
C. generator level (recomputes K2, K3, K4 on --generators frames: node_dt 3, nq 4, panel 3 (Gamma < 0.95) or 2):
       slowest-damped eigenvalue of K2 and of K2+K3+K4 at nodes t >= 6, and its first-order shift  d = l^T (K3+K4) r.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ..bath import normal_modes
from ..cases import parse_case_id
from ..frame import make_frame, polaron_filter
from ..tcl_higher import K_nodes
from .common import FIGS_DIR


def auc(x, y):
    from scipy.stats import mannwhitneyu
    a, b = x[y == 1], x[y == 0]
    return np.nan if len(a) < 5 or len(b) < 5 else mannwhitneyu(a, b).statistic / (len(a) * len(b))


def main(argv=None):
    import pandas as pd
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=FIGS_DIR); ap.add_argument("--generators", type=int, default=0); A = ap.parse_args(argv)
    fr = []
    for f in sorted(glob.glob(os.path.join(A.data, "csv", "*_frame_errors.csv"))):
        key = os.path.basename(f).split("_frame_errors")[0]; p = parse_case_id(key)
        if p.Gamma < 0.05:
            continue
        d = pd.read_csv(f); d["case"] = key; d["beta"] = p.beta; d["Gamma"] = p.Gamma
        d["regime"] = "activationless" if abs(p.omega_DA - p.Er) < 1e-9 else ("normal" if p.omega_DA < p.Er else "inverted")
        fr.append(d)
    D = pd.concat(fr, ignore_index=True); D["st2"] = D.stable_TCL2.astype(bool); D["st4"] = D.stable_TCL234.astype(bool); D["stK4"] = D.stable_K4.astype(bool)
    D["cell"] = (D.regime == "activationless") & (D.beta >= 2)
    D["r4bin"] = pd.cut(D.r4, [0, .25, .5, .75, 1, 1.5, 2, 3, np.inf])
    pd.set_option("display.width", 220)

    print("=== A1. stability against r4 (all frames with TCL2 stable) ===")
    S = D[D.st2]
    print(S.groupby("r4bin", observed=True).agg(frames=("r4", "size"), TCL234_stable=("st4", "mean"), K4_stable=("stK4", "mean")).round(3).to_string())
    print("AUC of r4 for TCL234 unstable: %.3f" % auc(S.r4.values, (~S.st4).astype(int).values))
    for c in (0.5, 0.75, 1.0):
        print("  r4>%.2f: unstable fraction %.2f | r4<=%.2f: %.2f" % (c, (~S[S.r4 > c].st4).mean(), c, (~S[S.r4 <= c].st4).mean()))

    B2 = D[D.st2 & D.st4].copy(); B2["ratio"] = B2.err_TCL234 / B2.err_TCL2; B2["worse"] = (B2.ratio > 1).astype(int)
    print("\n=== A2. frames stable for both: TCL234 worse than TCL2, against r4 ===")
    print(B2.groupby("r4bin", observed=True).agg(frames=("ratio", "size"), worse=("worse", "mean"), med_ratio=("ratio", "median")).round(3).to_string())
    print("AUC of r4 for 'worse': all %.3f | inside cell %.3f | outside cell %.3f" % (auc(B2.r4.values, B2.worse.values), auc(B2[B2.cell].r4.values, B2[B2.cell].worse.values),
                                                                                   auc(B2[~B2.cell].r4.values, B2[~B2.cell].worse.values)))
    print("AUC of the cell indicator for 'worse': %.3f" % auc(B2.cell.astype(float).values, B2.worse.values))
    print("\n--- cell effect at fixed r4 (fraction worse; frames) ---")
    print(B2.groupby(["r4bin", "cell"], observed=True).agg(n=("worse", "size"), worse=("worse", "mean")).round(3).unstack("cell").to_string())
    B2.round(5).to_csv(os.path.join(A.data, "mechanism_A_frames.csv"), index=False)

    print("\n=== B. residual bath weight at the transformed gap (activationless, beta>=2 cell) ===")
    rows = []
    for key, d in B2[B2.cell].groupby("case"):
        p = parse_case_id(key); m = normal_modes(p); om, R = m.omega, m.R
        for r in d.itertuples():
            frm = make_frame(p, r.alpha, r.omega_h); gap = frm.E[1] - frm.E[0]; f = polaron_filter(om, r.alpha, r.omega_h)
            cth = np.ones_like(om) if np.isinf(p.beta) else 1 / np.tanh(p.beta * om / 2); w = (om * R) ** 2 * (1 - f) ** 2 * cth
            row = dict(case=key, alpha=r.alpha, omega_h=r.omega_h, ratio=r.ratio, r4=r.r4, gap=gap)
            for s in (0.05, 0.1, 0.2):
                row[f"S{s}"] = np.sum(w * np.exp(-(om - gap) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)))
            rows.append(row)
    Bd = pd.DataFrame(rows)
    if len(Bd) > 20:
        for s in (0.05, 0.1, 0.2):
            x = np.log(Bd[f"S{s}"] + 1e-30)
            print("sigma=%.2f  Spearman(ratio, log S_res): pooled %+.2f | partial on r4 (residuals) %+.2f" % (
                s, spearmanr(Bd.ratio, x)[0], spearmanr(Bd.ratio - np.polyval(np.polyfit(Bd.r4, Bd.ratio, 1), Bd.r4), x - np.polyval(np.polyfit(Bd.r4, x, 1), Bd.r4))[0]))
        print("Spearman(ratio, gap) %+.2f   Spearman(r4, log S_res(0.1)) %+.2f" % (spearmanr(Bd.ratio, Bd.gap)[0], spearmanr(Bd.r4, np.log(Bd["S0.1"] + 1e-30))[0]))
        Bd.round(6).to_csv(os.path.join(A.data, "mechanism_B_frames.csv"), index=False)

    if A.generators > 0:
        print("\n=== C. generator spectrum (K2 against K2+K3+K4) ===")
        cell = D[D.cell & D.st2]; pick = []
        for key, d in cell.groupby("case"):
            bb = B2[B2.case == key]
            if len(bb):
                pick += [(key, "worst", bb.loc[bb.ratio.idxmax()]), (key, "best", bb.loc[bb.ratio.idxmin()])]
            dv = d[~d.st4]
            if len(dv):
                pick.append((key, "divergent", dv.iloc[(dv.r4 - dv.r4.median()).abs().argmin()]))
        out = []
        for key, kind, r in pick[:A.generators]:
            p = parse_case_id(key)
            kn = K_nodes(make_frame(p, r.alpha, r.omega_h), 24.0, 3.0, 4, 3.0 if p.Gamma < 0.95 else 2.0)
            Kd = {"t": kn.t, "K2": kn.K[2], "K3": kn.K[3], "K4": kn.K[4]}; res = {}
            for lab, K in (("K2", Kd["K2"]), ("K234", Kd["K2"] + Kd["K3"] + Kd["K4"])):
                m = []
                for i, t in enumerate(Kd["t"]):
                    if t < 6:
                        continue
                    ev = np.linalg.eigvals(K[i]); ev = ev[np.abs(ev) > 1e-8 * max(1, np.abs(ev).max())]; m.append(ev.real.max() if len(ev) else np.nan)
                res[lab] = np.array(m)
            lam, dd = [], []
            for i, t in enumerate(Kd["t"]):
                if t < 6:
                    continue
                ev, VR = np.linalg.eig(Kd["K2"][i]); ok = np.abs(ev) > 1e-8 * max(1, np.abs(ev).max())
                if not ok.any():
                    continue
                idx = np.where(ok)[0][np.argmax(ev.real[ok])]; VL = np.linalg.inv(VR)
                lam.append(ev[idx].real); dd.append((VL[idx] @ (Kd["K3"][i] + Kd["K4"][i]) @ VR[:, idx]).real)
            lam, dd = np.array(lam), np.array(dd)
            out.append(dict(case=key, kind=kind, alpha=r.alpha, omega_h=r.omega_h, r4=r.r4, lam2_re_t24=lam[-1], d34_re_t24=dd[-1],
                            r_gamma_t24=dd[-1] / abs(lam[-1]) if lam[-1] else np.nan, r_gamma_median=np.nanmedian(dd / np.abs(lam)),
                            maxRe_K2_t24=res["K2"][-1], maxRe_K234_t24=res["K234"][-1], maxRe_K234_median=np.nanmedian(res["K234"]),
                            frac_nodes_growing_K234=np.mean(res["K234"] > 1e-6)))
            print(pd.DataFrame(out[-1:]).round(4).to_string(header=len(out) == 1, index=False), flush=True)
        pd.DataFrame(out).round(6).to_csv(os.path.join(A.data, "mechanism_C_generators.csv"), index=False)
    print("\nwritten", os.path.join(A.data, "mechanism_*.csv"))


if __name__ == "__main__":
    main()
