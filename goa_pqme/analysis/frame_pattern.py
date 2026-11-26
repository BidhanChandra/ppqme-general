"""Where does TCL234 fail?  Case-level and frame-level patterns, from the frame-error CSVs (no new dynamics).

  python -m goa_pqme.analysis.frame_pattern [--data goa_figs_v2]   -> failure_pattern_cases.csv, failure_pattern_frames.csv
  python -m goa_pqme.analysis.frame_pattern --figure [--data goa_figs_v2] [--out pattern]   -> fig_failure_pattern.{pdf,png}

Per case: regime (normal / activationless / inverted from omega_DA vs E_r), beta, stable fraction, oracle ratio, and, over the
frames stable for both methods, the fraction where TCL234 beats TCL2, the median error ratio, and the dependence on the
transformed splitting Delta~ = E_1 - E_0 of H~_S and the mixing 2 Gamma B / Delta~.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from ..cases import parse_case_id
from ..frame import make_frame
from .common import FIGS_DIR


def pattern(data):
    import pandas as pd
    from scipy.stats import spearmanr
    rows, frames = [], []
    for f in sorted(glob.glob(os.path.join(data, "csv", "*_frame_errors.csv"))):
        key = os.path.basename(f).split("_frame_errors")[0]
        p = parse_case_id(key)
        if p.Gamma < 0.05:
            continue
        gam, od, b, Er = p.Gamma, p.omega_DA, p.beta, p.Er
        reg = "activationless" if abs(od - Er) < 1e-9 else ("normal" if od < Er else "inverted")
        fe = pd.read_csv(f); both = fe[fe.stable_TCL2.astype(bool) & fe.stable_TCL234.astype(bool)].copy()
        gap, mix = [], []
        for r in both.itertuples():
            fr = make_frame(p, r.alpha, r.omega_h); d = fr.E[1] - fr.E[0]; gap.append(d); mix.append(2 * gam * fr.B / d)
        both["gap"], both["mix"] = gap, mix; both["ratio"] = both.err_TCL234 / both.err_TCL2; both["case"] = key; both["regime"] = reg
        both["beta"] = b; both["Gamma"] = gam; frames.append(both)
        s2, s4 = fe[fe.stable_TCL2.astype(bool)], fe[fe.stable_TCL234.astype(bool)]
        rows.append(dict(case=key, Gamma=gam, w_DA=od, E_r=Er, regime=reg, beta=b, stable234=fe.stable_TCL234.astype(bool).mean(),
                         oracle2=s2.err_TCL2.min(), oracle234=s4.err_TCL234.min(), oracle_ratio=s4.err_TCL234.min() / s2.err_TCL2.min(),
                         frames_better=(both.ratio < 1).mean(), median_ratio=both.ratio.median(), median_gap=np.median(gap), median_mix=np.median(mix)))
    C = pd.DataFrame(rows).sort_values(["regime", "beta"], ascending=[True, False]); pd.set_option("display.width", 250)
    print("=== per case (frames stable for both methods) ===\n", C[["case", "regime", "beta", "stable234", "oracle_ratio", "frames_better", "median_ratio", "median_gap", "median_mix"]].round(3).to_string(index=False))
    F = pd.concat(frames); F["bclass"] = np.where(F.beta >= 2, "beta>=2", "beta<=1")
    print("\n=== pooled by regime and temperature class: fraction of frames where TCL234 is better, median err ratio ===")
    print(F.groupby(["regime", "bclass"]).agg(frames=("ratio", "size"), better=("ratio", lambda x: (x < 1).mean()), med_ratio=("ratio", "median")).round(3).to_string())
    print("\n=== within activationless cases: dependence on splitting (frame level) ===")
    R = F[F.regime == "activationless"].copy(); R["gap_bin"] = pd.qcut(R.gap, 4, duplicates="drop")
    print(R.groupby(["bclass", "gap_bin"], observed=True).agg(frames=("ratio", "size"), better=("ratio", lambda x: (x < 1).mean()), med_ratio=("ratio", "median")).round(3).to_string())
    print("\nSpearman(err ratio, log splitting) | (err ratio, mixing)  per case:")
    for k, d in F.groupby("case"):
        if len(d) > 10:
            print("  %-46s %+.2f  %+.2f" % (k[:46], spearmanr(d.ratio, np.log(d.gap))[0], spearmanr(d.ratio, d.mix)[0]))
    C.round(5).to_csv(os.path.join(data, "failure_pattern_cases.csv"), index=False); F.round(5).to_csv(os.path.join(data, "failure_pattern_frames.csv"), index=False)


def figure(data, out):
    """Four-panel failure-pattern figure from failure_pattern_cases.csv / _frames.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from ..cases import case_labels
    os.makedirs(out, exist_ok=True)
    COL = {"normal": "#2a78d6", "activationless": "#eb6834", "inverted": "#1baf7a"}; MK = {"normal": "o", "activationless": "s", "inverted": "^"}
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    cs = pd.read_csv(os.path.join(data, "failure_pattern_cases.csv")); F = pd.read_csv(os.path.join(data, "failure_pattern_frames.csv")); cs["beta"] = cs.beta.astype(float)
    F["bclass"] = np.where(F.beta >= 2, "beta>=2", "beta<=1")
    g = lambda d: pd.Series(dict(frames=len(d), better=(d.ratio < 1).mean(), med_ratio=d.ratio.median()))
    pooled = F.groupby(["regime", "bclass"]).apply(g, include_groups=False).reset_index()
    R = F[F.regime == "activationless"].copy(); R["bin"] = pd.qcut(R.gap, 4, duplicates="drop")
    bins = R.groupby(["bclass", "bin"], observed=True).apply(g, include_groups=False).reset_index(); bins["bin"] = bins["bin"].map(lambda b: "%.2f-%.2f" % (b.left, b.right))
    lab = case_labels(list(cs.case)); cs["lab"] = cs.case.map(lambda k: lab.get(k, k[:8]))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    a = ax[0, 0]; order = [("normal", "beta>=2"), ("inverted", "beta>=2"), ("activationless", "beta>=2"), ("normal", "beta<=1"), ("activationless", "beta<=1")]
    pn = {"beta>=2": r"$\beta\geq2$", "beta<=1": r"$\beta\leq1$"}; y = np.arange(len(order))[::-1]
    for yy, (rg, bc) in zip(y, order):
        sel = pooled[(pooled.regime == rg) & (pooled.bclass == bc)]
        if not len(sel):
            continue
        r = sel.iloc[0]
        a.barh(yy, 100 * r.better, height=0.62, color=COL[rg], edgecolor="none")
        a.text(100 * r.better + 1.5, yy, "%.0f%%  (n=%d)" % (100 * r.better, r.frames), va="center", fontsize=7.5, color=INK)
    a.axvline(50, color=INK2, lw=0.8, ls=(0, (3, 3))); a.set_yticks(y); a.set_yticklabels(["%s, %s" % (rg, pn[bc]) for rg, bc in order]); a.set_xlim(0, 125)
    a.set_xticks([0, 20, 40, 60, 80, 100]); a.set_xlabel("frames where TCL234 beats TCL2 (%)"); a.set_title("(a) Regime and temperature", loc="left", fontsize=9, color=INK)
    b = ax[0, 1]; bl = [np.inf, 10, 5, 2, 1, 0.5]; xp = {v: i for i, v in enumerate(bl)}
    for rg in COL:
        d = cs[cs.regime == rg]; xs = np.array([xp[v] for v in d.beta], float) + {"normal": -0.17, "activationless": 0.0, "inverted": 0.17}[rg]
        b.scatter(xs, d.median_ratio, s=18 + 90 * d.stable234 ** 2, marker=MK[rg], color=COL[rg], edgecolor="#fcfcfb", lw=0.8, label=rg, zorder=3)
        for x_, y_, l_ in zip(xs, d.median_ratio, d.lab):
            if rg == "activationless" or y_ > 0.9:
                b.annotate(l_, (x_, y_), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=INK2)
    b.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3))); b.set_yscale("log"); b.set_ylim(0.22, 2.8); b.set_yticks([0.25, 0.5, 1, 2]); b.set_yticklabels(["0.25", "0.5", "1", "2"])
    b.set_xticks(range(6)); b.set_xticklabels([r"$\infty$", "10", "5", "2", "1", "0.5"]); b.set_xlabel(r"inverse temperature $\beta$")
    b.set_ylabel("median error ratio TCL234 / TCL2\n(frames stable for both)"); b.legend(frameon=False, fontsize=7, loc="upper right", handletextpad=0.3)
    b.set_title("(b) Cases (marker size = stable fraction)", loc="left", fontsize=9, color=INK)
    c = ax[1, 0]
    for bc, mk, ls, lb in (("beta>=2", "s", "-", r"activationless, $\beta\geq2$"), ("beta<=1", "D", (0, (3, 2)), r"activationless, $\beta\leq1$")):
        d = bins[bins.bclass == bc]; x = np.arange(len(d))
        c.plot(x, 100 * d.better, marker=mk, ms=6, lw=1.6, ls=ls, color=COL["activationless"], mfc=COL["activationless"] if bc == "beta>=2" else "#fcfcfb", mew=1.4, label=lb)
        for xi, (bt, n) in enumerate(zip(d.better, d.frames)):
            c.annotate("n=%d" % n, (xi, 100 * bt), xytext=((16 if xi == 0 else 0) if bc == "beta>=2" else 0, (-8 if xi == 0 else -15) if bc == "beta>=2" else 8),
                       textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    c.axhline(50, color=INK2, lw=0.8, ls=(0, (3, 3))); c.set_ylim(0, 112); c.set_xticks(range(4))
    c.set_xticklabels(bins[bins.bclass == "beta>=2"]["bin"].tolist() if (bins.bclass == "beta>=2").any() else [])
    c.set_xlabel(r"transformed splitting $\tilde\Delta$ of $\widetilde H_S$  (light $\rightarrow$ strong dressing)"); c.set_ylabel("frames where TCL234 beats TCL2 (%)")
    c.legend(frameon=False, fontsize=7, loc="upper right"); c.set_title("(c) At $\\omega_{DA}=E_r$: dependence on dressing", loc="left", fontsize=9, color=INK)
    d_ = ax[1, 1]
    for rg in COL:
        d = cs[cs.regime == rg]; d_.scatter(d.stable234, d.median_ratio, s=34, marker=MK[rg], color=COL[rg], edgecolor="#fcfcfb", lw=0.8, zorder=3)
        for x_, y_, l_ in zip(d.stable234, d.median_ratio, d.lab):
            if y_ > 0.9:
                d_.annotate(l_, (x_, y_), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    d_.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3))); d_.set_yscale("log"); d_.set_ylim(0.22, 2.8); d_.set_yticks([0.25, 0.5, 1, 2]); d_.set_yticklabels(["0.25", "0.5", "1", "2"])
    d_.set_xlabel("TCL234 stable fraction"); d_.set_ylabel("median error ratio TCL234 / TCL2"); d_.set_title("(d) Failures are not low-stability cases", loc="left", fontsize=9, color=INK)
    for x in ax.ravel():
        x.set_axisbelow(True); x.minorticks_off(); x.grid(axis="y" if x is not ax[0, 0] else "x", color=GRID, lw=0.6, zorder=0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig_failure_pattern.%s" % ext), dpi=220)
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=FIGS_DIR); ap.add_argument("--figure", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.figure:
        figure(a.data, a.out or os.path.join(a.data, "figs"))
    else:
        pattern(a.data)


if __name__ == "__main__":
    main()
