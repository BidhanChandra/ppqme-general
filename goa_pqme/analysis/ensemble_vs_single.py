"""Ensemble (reference-free median) versus single frames chosen without the reference.

  python -m goa_pqme.analysis.ensemble_vs_single [--data goa_figs_v2] [--out goa_figs_v2/figs]
Needs <data>/csv/*_frame_errors.csv and <data>/paper_numbers_per_case.csv (from analysis.paper_numbers).
Single-frame rules: laboratory frame (largest B), fully dressed frame (smallest B), smallest r4, typical, oracle.
Writes fig_ensemble_vs_single.{pdf,png} and ensemble_vs_single_table.csv; prints means and paired statistics.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ..cases import case_labels, parse_case_id
from ..frame import make_frame
from .common import FIGS_DIR


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=FIGS_DIR); ap.add_argument("--out", default=None); A = ap.parse_args(argv)
    out = A.out or os.path.join(A.data, "figs"); os.makedirs(out, exist_ok=True)
    files = {os.path.basename(f).split("_frame_errors")[0]: f for f in glob.glob(os.path.join(A.data, "csv", "*_frame_errors.csv"))}
    LAB = case_labels(files)
    LAB = {k: LAB[k] for k in sorted(LAB, key=lambda k: (LAB[k][0] == "H", int(LAB[k][1:])))}
    RFt = pd.read_csv(os.path.join(A.data, "paper_numbers_per_case.csv"), index_col=0)
    rows, sel = [], []
    for key, L in LAB.items():
        d = pd.read_csv(files[key])
        if "B" not in d:
            p = parse_case_id(key)
            d["B"] = [make_frame(p, r.alpha, r.omega_h).B for r in d.itertuples()]
        r = dict(case=L)
        for m, k in (("TCL2", "2"), ("TCL234", "234")):
            s = d[d["stable_" + m].astype(bool) & np.isfinite(d.r4) & np.isfinite(d["err_" + m])]
            r["RF" + k] = RFt.loc[L, "RF_" + m]
            r["lab" + k] = s.loc[s.B.idxmax(), "err_" + m]; r["dress" + k] = s.loc[s.B.idxmin(), "err_" + m]
            j = s.r4.idxmin(); r["minr4_" + k] = s.loc[j, "err_" + m]; r["typ" + k] = s["err_" + m].median(); r["orc" + k] = s["err_" + m].min()
            if k == "234":
                sel.append(dict(case=L, B=s.loc[j, "B"], r4=s.loc[j, "r4"]))
        rows.append(r)
    R = pd.DataFrame(rows).set_index("case").astype(float); S = pd.DataFrame(sel).set_index("case")
    R = R.loc[sorted(R.index, key=lambda c: (c[0] != "C", int(c[1:])))]; S = S.loc[R.index]
    devL = [c for c in R.index if c[0] == "C"]; heldL = [c for c in R.index if c[0] == "H"]
    R.round(5).to_csv(os.path.join(out, "ensemble_vs_single_table.csv"))
    print("mean error (dev / held / all)")
    for k, nm in (("2", "TCL2"), ("234", "TCL234")):
        for c, lab in (("RF", "ensemble RF"), ("minr4_", "min-r4 frame"), ("lab", "lab frame"), ("dress", "dressed frame"), ("typ", "typical"), ("orc", "oracle")):
            print("%-7s %-14s %.4f / %.4f / %.4f" % (nm, lab, R.loc[devL, c + k].mean(), R.loc[heldL, c + k].mean(), R[c + k].mean()))
    try:
        from scipy import stats
        rng = np.random.default_rng(0)
        for k in ("2", "234"):
            dd = (R["minr4_" + k] - R["RF" + k]).values; bs = np.percentile([rng.choice(dd, len(dd)).mean() for _ in range(20000)], [2.5, 97.5])
            print("TCL%s: mean(min-r4 - RF) %+.4f  95%% CI [%+.4f, %+.4f]  Wilcoxon p = %.2f" % (k, dd.mean(), bs[0], bs[1], stats.wilcoxon(dd).pvalue))
    except Exception as e:
        print("paired statistics skipped:", e)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    cE, cM, cL, cD, cO = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"
    fig = plt.figure(figsize=(7.4, 5.8)); gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.5, wspace=0.75)
    ax = fig.add_subplot(gs[0, :]); x = np.arange(len(R))
    for c, mk, col, lb, kw in (("dress234", "x", cD, "fully dressed frame", {}), ("lab234", "^", cL, "laboratory frame", {}), ("RF234", "o", cE, "ensemble median (RF)", {}),
                              ("minr4_234", "o", cM, "single frame, smallest $r_4$", dict(mfc="none")), ("orc234", "_", cO, "best frame (oracle)", {})):
        ax.plot(x, R[c], ls="none", marker=mk, color=col, ms=6 if mk != "_" else 9, mew=1.2, label=lb, **kw)
    ax.axvline(len(devL) - 0.5, color="#bbbbbb", lw=0.6); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(R.index); ax.set_ylabel("error vs TEMPO (TCL234)")
    yt = ax.get_ylim()[1]
    ax.text(len(devL) / 2 - 0.5, yt, "development", ha="center", va="top", fontsize=7, color="#555555")
    ax.text(len(devL) + len(heldL) / 2 - 0.5, yt, "held-out", ha="center", va="top", fontsize=7, color="#555555")
    ax.legend(ncol=5, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.0); ax.set_title("(a)", loc="left", fontsize=8, pad=18)
    ax = fig.add_subplot(gs[1, 0]); lim = [min(R.RF234.min(), R.minr4_234.min()) * 0.7, max(R.RF234.max(), R.minr4_234.max()) * 1.4]
    ax.plot(lim, lim, color="#999999", lw=0.7, zorder=0)
    for L in R.index:
        b = S.loc[L, "B"]; mk = "o" if b > 0.9 else ("^" if b < 0.1 else "s")
        ax.plot(R.loc[L, "RF234"], R.loc[L, "minr4_234"], marker=mk, color=cM, mfc=cM if b > 0.9 else "none", ms=5.5, mew=1.1, ls="none")
        if L in ("C5", "C10", "H2", "H4", "C7"):
            ax.annotate(L, (R.loc[L, "RF234"], R.loc[L, "minr4_234"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("ensemble median error"); ax.set_ylabel("smallest-$r_4$ frame error")
    ax.text(0.04, 0.96, "filled: nearly undressed ($B>0.9$)\nsquare: partial\ntriangle: fully dressed ($B<0.1$)", transform=ax.transAxes, va="top", fontsize=6.5, color="#444444")
    ax.text(0.97, 0.04, "below line:\nsingle frame better", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#444444"); ax.set_title("(b)", loc="left", fontsize=8)
    ax = fig.add_subplot(gs[1, 1]); names = [("RF", "ensemble median", cE), ("minr4_", "smallest-$r_4$ frame", cM), ("typ", "typical frame", cO), ("lab", "laboratory frame", cL), ("dress", "fully dressed frame", cD)]
    y = np.arange(len(names))[::-1]
    for (c, lb, col), yy in zip(names, y):
        ax.plot([R[c + "2"].mean()], [yy + 0.14], marker="s", color=col, mfc="none", ms=5.5, ls="none"); ax.plot([R[c + "234"].mean()], [yy - 0.14], marker="o", color=col, ms=5.5, ls="none")
    ax.set_yticks(y); ax.set_yticklabels([n[1] for n in names]); ax.set_xlabel("mean error over %d cases" % len(R)); ax.set_xlim(0, None)
    ax.plot([], [], "s", color="#444444", mfc="none", label="TCL2"); ax.plot([], [], "o", color="#444444", label="TCL234"); ax.legend(frameon=False, fontsize=7, loc="upper right"); ax.set_title("(c)", loc="left", fontsize=8)
    fig.savefig(os.path.join(out, "fig_ensemble_vs_single.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(out, "fig_ensemble_vs_single.png"), dpi=300, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
