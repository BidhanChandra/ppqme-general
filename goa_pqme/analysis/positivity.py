"""Positivity audits of the TRANSFORMED-FRAME reduced state rho~_S(t), from a population scan (dP) and coherences (B rho~_DA).

For a unit-trace two-state matrix the eigenvalues are (1 +- sqrt(dP^2 + 4|rho~_DA|^2))/2, so
    lambda_min(t) = (1 - sqrt(dP^2 + 4 |rho~_DA|^2)) / 2,        rho~_DA = (B rho~_DA) / B   (the Q1 term is not used).

  python -m goa_pqme.analysis.positivity audit   --scan work/scan [--coh DIR] [--tempo-dir notebooks/csv] [--tol 1e-5]
  python -m goa_pqme.analysis.positivity checked --scan work/scan [--coh DIR] [--tol 1e-3]     (explicit nesting / data checks)
  python -m goa_pqme.analysis.positivity sweep   --scan work/scan [--coh DIR] [--taus 1e-12,...,1e-3]

audit:   screens  paper (|dP| <= 1.6) | |dP| <= 1 | lambda_min >= -tol : kept fractions and RF errors TCL2 -> TCL234.
checked: the same with strict / tolerant positivity, asserting the nesting of the screens; writes a per-frame table.
sweep:   tolerance sweep, method-specific and common-frame RF errors, worst violation among paper-screen frames.
--coh defaults to --scan (new scans hold dP and coherence in one file; legacy data: --scan work/scan_new --coh work/scan_coh).
Outputs go to --out (default: current folder): positivity_audit.csv, positivity_audit_checked.csv (+ _frames.csv), positivity_sweep.csv.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ..cases import DEV_PREFIXES, HELD_PREFIXES
from ..io import load_frame
from .common import TEMPO_DIR, scan_reference

SETS = (("dev", DEV_PREFIXES), ("held", HELD_PREFIXES))


def _cids(scan):
    return sorted({os.path.basename(f).split("__a")[0] for f in glob.glob(os.path.join(scan, "curves", "*.npz"))})


def _cid_of(prefix, cids, strict=False):
    s = prefix.replace(".", "p")
    m = [c for c in cids if c == s or c.startswith(s + "_")] if strict else [c for c in cids if (c + "_").startswith(s + "_") or c.startswith(s + "_")]
    if strict and len(m) != 1:
        raise RuntimeError(f"Case prefix {prefix}: expected one case ID, got {m}")
    return m[0]


def _pair(f, coh_dir):
    """(t, dP{TCL2,TCL234}, coherence t, coherence {TCL2_B,...}, B) for one population frame file."""
    z = load_frame(f)
    g = os.path.join(coh_dir, "curves", os.path.basename(f))
    if not os.path.exists(g):
        return None
    c = z if (os.path.abspath(g) == os.path.abspath(f)) else load_frame(g)
    if not c["coherence"]:
        return None
    return z["t"], z["dP"], c["t"], c["coherence"], float(c["B"])


def audit(scan, coh, tempo_dir, tol, out):
    import pandas as pd
    cids = _cids(scan)
    SCR = ["paper (|dP|<=1.6)", "|dP|<=1", "full positivity"]
    rows = {}
    for lab, plist in SETS:
        for p in plist:
            cid = _cid_of(p, cids); ref, _ = scan_reference(tempo_dir, cid)
            acc = {m: {s: [] for s in SCR} for m in ("TCL2", "TCL234")}; n = 0; t = None
            for f in glob.glob(os.path.join(scan, "curves", cid + "__a*.npz")):
                pr = _pair(f, coh)
                if pr is None:
                    continue
                t, dPs, ct, cc, B = pr; n += 1
                for m in acc:
                    dP = dPs[m]; rho = np.interp(t, ct, np.real(cc[m + "_B"])) + 1j * np.interp(t, ct, np.imag(cc[m + "_B"])); rho = rho / B
                    fin = bool(np.all(np.isfinite(dP)) and np.all(np.isfinite(rho)))
                    lam = (1 - np.sqrt(dP ** 2 + 4 * np.abs(rho) ** 2)) / 2 if fin else None
                    acc[m][SCR[0]].append((fin and np.abs(dP).max() <= 1.6, dP)); acc[m][SCR[1]].append((fin and np.abs(dP).max() <= 1.0, dP))
                    acc[m][SCR[2]].append((fin and lam.min() >= -tol, dP))
            mk = t <= ref.t.iloc[-1] + 1e-9; rd = np.interp(t, ref.t, ref.dP)
            r = dict(set=lab, n=n)
            for m in acc:
                for s in SCR:
                    Y = [d for ok, d in acc[m][s] if ok]; r[f"{m}|{s}|frac"] = len(Y) / max(n, 1)
                    r[f"{m}|{s}|err"] = 0.5 * np.sqrt(np.mean((np.median(Y, 0)[mk] - rd[mk]) ** 2)) if Y else np.nan
            rows[p] = r
    T = pd.DataFrame(rows).T; pd.set_option("display.width", 250)
    print("lambda_min tolerance %.0e; frames per case: %d" % (tol, int(T.n.median())))
    for s in SCR:
        print("\n=== screen:", s, "===")
        for nm, sel in (("dev", T.set == "dev"), ("held", T.set == "held"), ("all", T.set != "")):
            e2, e4 = T.loc[sel, f"TCL2|{s}|err"].astype(float), T.loc[sel, f"TCL234|{s}|err"].astype(float)
            f2, f4 = T.loc[sel, f"TCL2|{s}|frac"].astype(float), T.loc[sel, f"TCL234|{s}|frac"].astype(float)
            print("%-5s RF error %.4f -> %.4f (%+.0f%%; TCL234 better in %d of %d) | kept frames TCL2 %.2f, TCL234 %.2f | cases with no kept TCL234 frame: %d" % (
                nm, e2.mean(), e4.mean(), 100 * (e4.mean() / e2.mean() - 1), int((e4 < e2).sum()), int(sel.sum()), f2.mean(), f4.mean(), int(e4.isna().sum())))
    path = os.path.join(out, "positivity_audit.csv"); T.to_csv(path); print("\nwritten", path)


def checked(scan, coh, tempo_dir, tol, out):
    import pandas as pd
    SCREENS = ("paper", "population", "positive_strict", "positive_tol")
    cids = _cids(scan)
    if not cids:
        raise RuntimeError("No population .npz files found")
    records, rows = [], []
    for group, names in SETS:
        for prefix in names:
            cid = _cid_of(prefix, cids, strict=True)
            ref, fs = scan_reference(tempo_dir, cid)
            if len(fs) != 1:
                raise RuntimeError(f"{cid}: expected exactly one TEMPO reference, got {fs}")
            rt, rd = np.asarray(ref["t"], float), np.asarray(ref["dP"], float)
            if len(rt) == 0 or np.any(np.diff(rt) <= 0):
                raise RuntimeError(f"{fs[0]} has invalid time grid")
            files = sorted(glob.glob(os.path.join(scan, "curves", cid + "__a*.npz")))
            if not files:
                raise RuntimeError(f"{cid}: no frames")
            traj = {m: {s: [] for s in SCREENS} for m in ("TCL2", "TCL234")}
            t0 = None
            for path in files:
                pr = _pair(path, coh)
                if pr is None:
                    raise RuntimeError(f"Missing matched coherence frame for {path}")
                t, dPs, ct, cc, B = pr
                t, ct = np.asarray(t, float), np.asarray(ct, float)
                if t.ndim != 1 or ct.ndim != 1 or np.any(np.diff(t) <= 0) or np.any(np.diff(ct) <= 0):
                    raise RuntimeError(f"Invalid time grid: {path}")
                if t[0] < ct[0] - 1e-10 or t[-1] > ct[-1] + 1e-10:
                    raise RuntimeError(f"Coherence time grid does not cover population interval: {path}")
                if t0 is None:
                    t0 = t
                elif len(t) != len(t0) or not np.allclose(t, t0, rtol=0, atol=1e-9):
                    raise RuntimeError(f"Inconsistent population time grids in {cid}")
                if not np.isfinite(B) or B <= 0:
                    raise RuntimeError(f"Invalid B factor in {path}: {B}")
                for method in ("TCL2", "TCL234"):
                    dP = np.asarray(dPs[method], float); cb = np.asarray(cc[method + "_B"], complex)
                    if dP.shape != t.shape or cb.shape != ct.shape:
                        raise RuntimeError(f"Unexpected trajectory shape in {path}, {method}")
                    cohv = (np.interp(t, ct, cb.real) + 1j * np.interp(t, ct, cb.imag)) / B
                    fin = np.all(np.isfinite(dP)) and np.all(np.isfinite(cohv))
                    if fin:
                        minlam = float(np.min(0.5 * (1 - np.sqrt(dP * dP + 4 * np.abs(cohv) ** 2)))); maxpop = float(np.max(np.abs(dP)))
                        paper, population, positive_strict, positive_tol = maxpop <= 1.6, maxpop <= 1.0, minlam >= 0.0, minlam >= -tol
                        if positive_strict and not population:
                            raise AssertionError(f"STRICT nesting failed: {path} {method}")
                        if positive_tol and maxpop > 1 + 2 * tol + 1e-10:
                            raise AssertionError(f"TOLERANCE algebra failed: {path} {method}")
                        if positive_tol and not paper and tol <= 0.3:
                            raise AssertionError(f"Paper nesting failed: {path} {method}")
                    else:
                        minlam, maxpop = np.nan, np.nan
                        paper = population = positive_strict = positive_tol = False
                    flags = dict(paper=paper, population=population, positive_strict=positive_strict, positive_tol=positive_tol)
                    for crit, ok in flags.items():
                        if ok:
                            traj[method][crit].append(dP)
                    records.append(dict(case=prefix, group=group, frame=os.path.basename(path), method=method, B=B, finite=bool(fin),
                                        max_abs_dP=maxpop, min_lambda=minlam, **flags))
            if t0[-1] > rt[-1] + 1e-9 or t0[0] < rt[0] - 1e-9:
                raise RuntimeError(f"{cid}: TEMPO reference does not cover propagation grid")
            rdi = np.interp(t0, rt, rd); n = len(files)
            for method in traj:
                for crit in SCREENS:
                    sub = traj[method][crit]
                    e = float(0.5 * np.sqrt(np.mean((np.median(sub, axis=0) - rdi) ** 2))) if sub else np.nan
                    rows.append(dict(case=prefix, group=group, method=method, screen=crit, n_frames=n, n_kept=len(sub), fraction=len(sub) / n, rf_error=e))
    O = pd.DataFrame(rows); Fr = pd.DataFrame(records)
    p1, p2 = os.path.join(out, "positivity_audit_checked.csv"), os.path.join(out, "positivity_audit_frames.csv")
    O.to_csv(p1, index=False); Fr.to_csv(p2, index=False); print("Saved", p1, "and", p2)
    for crit in SCREENS:
        for group in ("dev", "held", "all"):
            sub = O[O.screen == crit]
            if group != "all":
                sub = sub[sub.group == group]
            means = sub.groupby("method")["rf_error"].mean(); fr = sub.groupby("method")["fraction"].mean()
            print(f'{crit:16s} {group:4s}  TCL2={means.get("TCL2", float("nan")):.5f}  TCL234={means.get("TCL234", float("nan")):.5f}  fractions={fr.to_dict()}')


def sweep(scan, coh, tempo_dir, taus, out):
    import pandas as pd
    cids = _cids(scan); PAPER = 1.6
    cases = []; worst_B = (np.inf, None)
    for lab, pl in SETS:
        for p in pl:
            cid = _cid_of(p, cids); ref, _ = scan_reference(tempo_dir, cid); fr = []; t = None
            for f in glob.glob(os.path.join(scan, "curves", cid + "__a*.npz")):
                pr = _pair(f, coh)
                if pr is None:
                    continue
                t, dPs, ct, cc, B = pr; rec = {}
                for m in ("TCL2", "TCL234"):
                    dP = dPs[m]; raw = cc[m + "_B"]; rho = (np.interp(t, ct, raw.real) + 1j * np.interp(t, ct, raw.imag)) / B
                    fin = bool(np.all(np.isfinite(dP)) and np.all(np.isfinite(rho)))
                    rec[m] = (dP, fin, (np.min((1 - np.sqrt(dP ** 2 + 4 * np.abs(rho) ** 2)) / 2) if fin else -np.inf), (fin and np.abs(dP).max() <= PAPER))
                if B < worst_B[0]:
                    worst_B = (B, (cid, os.path.basename(f), float(np.max(np.abs((raw / B) * B - raw) / np.maximum(np.abs(raw), 1e-300)))))
                fr.append(rec)
            mk = t <= ref.t.iloc[-1] + 1e-9; rd = np.interp(t, ref.t, ref.dP); cases.append((lab, p, fr, mk, rd))
    err = lambda Y, mk, rd: np.nan if not Y else 0.5 * np.sqrt(np.mean((np.median(Y, 0)[mk] - rd[mk]) ** 2))
    print("smallest B: %.1e (%s); max relative error of (raw/B)*B vs raw: %.1e\n" % (worst_B[0], worst_B[1][1], worst_B[1][2]))
    rows = []
    for tau in taus:
        R = []
        for lab, p, fr, mk, rd in cases:
            ok = lambda r, m: r[m][2] >= -tau
            e2 = err([r["TCL2"][0] for r in fr if ok(r, "TCL2")], mk, rd); e4 = err([r["TCL234"][0] for r in fr if ok(r, "TCL234")], mk, rd)
            both = [r for r in fr if ok(r, "TCL2") and ok(r, "TCL234")]; c2 = err([r["TCL2"][0] for r in both], mk, rd); c4 = err([r["TCL234"][0] for r in both], mk, rd)
            R.append(dict(set=lab, f2=np.mean([ok(r, "TCL2") for r in fr]), f4=np.mean([ok(r, "TCL234") for r in fr]), fb=len(both) / len(fr), e2=e2, e4=e4, c2=c2, c4=c4))
        R = pd.DataFrame(R)
        for nm, sel in (("dev", R.set == "dev"), ("held", R.set == "held"), ("all", R.set != "")):
            d = R[sel]
            rows.append(dict(tau="%.0e" % tau, set=nm, kept_TCL2=d.f2.mean(), kept_TCL234=d.f4.mean(), kept_both=d.fb.mean(),
                             specific=f"{d.e2.mean():.4f}->{d.e4.mean():.4f} ({100*(d.e4.mean()/d.e2.mean()-1):+.0f}%, better {int((d.e4<d.e2).sum())}/{len(d)})",
                             common=f"{d.c2.mean():.4f}->{d.c4.mean():.4f} ({100*(d.c4.mean()/d.c2.mean()-1):+.0f}%, better {int((d.c4<d.c2).sum())}/{len(d)})",
                             no_common_frames=int(d.c2.isna().sum())))
    T = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(T.to_string(index=False, float_format=lambda x: "%.2f" % x))
    print("\nWorst lambda_min among frames passing the paper screen (|dP|<=1.6), per method:")
    for m in ("TCL2", "TCL234"):
        w = [(r[m][2], p) for lab, p, fr, mk, rd in cases for r in fr if r[m][3]]; print("  %s: %.2e (%s)" % (m, min(w)[0], min(w)[1]))
    path = os.path.join(out, "positivity_sweep.csv"); T.to_csv(path, index=False); print("written", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["audit", "checked", "sweep"])
    ap.add_argument("--scan", required=True); ap.add_argument("--coh", default=None); ap.add_argument("--tempo-dir", default=TEMPO_DIR)
    ap.add_argument("--tol", type=float, default=None); ap.add_argument("--taus", default="1e-12,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    coh = a.coh or a.scan
    os.makedirs(a.out, exist_ok=True)
    if a.what == "audit":
        audit(a.scan, coh, a.tempo_dir, 1e-5 if a.tol is None else a.tol, a.out)
    elif a.what == "checked":
        tol = 1e-3 if a.tol is None else a.tol
        if tol < 0:
            ap.error("--tol must be nonnegative")
        checked(a.scan, coh, a.tempo_dir, tol, a.out)
    else:
        sweep(a.scan, coh, a.tempo_dir, [float(x) for x in a.taus.split(",")], a.out)


if __name__ == "__main__":
    main()
