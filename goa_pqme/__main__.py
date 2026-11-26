"""Command line:

    python -m goa_pqme info  G0p2_OD0_Binf_y0p75 [--alpha 2 --omega-h 0.3]
    python -m goa_pqme frame G0p2_OD0_Binf_y0p75 0.5 0.05 [--numerics production|converged] [--out DIR]
    python -m goa_pqme scan  --cases all|id1,id2 [--nproc 8] [--numerics converged] [--node-dt/--nq/--panel-h ...]
                             [--no-coherence] [--resolve] [--stride 2] [--grid paper|legacy] [--overwrite] [--only-stable-from SUMMARY] [--out DIR]
    python -m goa_pqme summary DIR                     rebuild DIR/summary.csv from the frame files
    python -m goa_pqme tempo G0p2_OD0_Binf_y0p75 [--dt 0.1 --tmax 24 --eps 1e-8] --out file.csv
Studies:  python -m goa_pqme.studies.{ladder,bias,tempo_conv,checks} --help
Analyses: python -m goa_pqme.analysis.<name> --help      Active learning: python -m goa_pqme.al.<name> --help
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import config
from .cases import case_id, get_case, load_cases, load_reference


def _numerics(args):
    n = {"production": config.PRODUCTION, "converged": config.CONVERGED}[args.numerics]
    kw = {k: v for k, v in (("tmax", args.tmax), ("node_dt", getattr(args, "node_dt", None)), ("nq", getattr(args, "nq", None)),
                            ("panel_h", getattr(args, "panel_h", None))) if v is not None}
    return n.with_(**kw)


def _add_numerics(p):
    p.add_argument("--numerics", default="converged", choices=["converged", "production"])
    p.add_argument("--tmax", type=float); p.add_argument("--node-dt", type=float); p.add_argument("--nq", type=int); p.add_argument("--panel-h", type=float)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="goa_pqme", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info"); a.add_argument("case"); a.add_argument("--alpha", type=float); a.add_argument("--omega-h", type=float)
    f = sub.add_parser("frame"); f.add_argument("case"); f.add_argument("alpha", type=float); f.add_argument("omega_h", type=float)
    _add_numerics(f); f.add_argument("--out")
    s = sub.add_parser("scan"); s.add_argument("--cases", default="all"); s.add_argument("--nproc", type=int, default=1); _add_numerics(s)
    s.add_argument("--out"); s.add_argument("--no-coherence", action="store_true"); s.add_argument("--resolve", action="store_true")
    s.add_argument("--alphas"); s.add_argument("--omega-hs"); s.add_argument("--stride", type=int, default=1)
    s.add_argument("--grid", default="paper", choices=["paper", "legacy"], help="paper: ALPHAS x OMEGAHS; legacy: first production grid")
    s.add_argument("--overwrite", action="store_true"); s.add_argument("--only-stable-from", default=None,
                                                                     help="summary.csv: run only frames stable for TCL2 or TCL234 there")
    m = sub.add_parser("summary"); m.add_argument("dir")
    t = sub.add_parser("tempo"); t.add_argument("case"); t.add_argument("--dt", type=float, default=0.1)
    t.add_argument("--tmax", type=float, default=24.0); t.add_argument("--eps", type=float); t.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "info":
        from .scan import print_model
        print_model(get_case(args.case), args.alpha, args.omega_h)
    elif args.cmd == "frame":
        from .metrics import half_rmse
        from .pipeline import run_frame
        p = get_case(args.case); num = _numerics(args)
        res = run_frame(p, args.alpha, args.omega_h, num)
        row = dict(case=args.case, alpha=args.alpha, omega_h=args.omega_h, r3=res.r3, r4=res.r4, seconds=round(res.seconds, 1))
        try:
            ref = load_reference(args.case)
            row.update({m_: half_rmse(res.t, y, ref.t.values, ref.dP.values) for m_, y in res.dP.items()})
        except FileNotFoundError:
            pass
        row.update({"max_" + m_: float(np.nanmax(np.abs(y))) for m_, y in res.dP.items()})
        if args.out:
            from .io import frame_filename, save_frame
            save_frame(frame_filename(args.out, args.case, args.alpha, args.omega_h), res, p, num)
        print(json.dumps(row))
    elif args.cmd == "scan":
        from .scan import scan_frames, stable_frames_from_summary
        ids = list(load_cases()) if args.cases == "all" else args.cases.split(",")
        A0, W0 = (config.ALPHAS, config.OMEGAHS) if args.grid == "paper" else (config.LEGACY_ALPHAS, config.LEGACY_OMEGAHS)
        al = [float(x) for x in args.alphas.split(",")] if args.alphas else list(A0)[::args.stride]
        ws = [float(x) for x in args.omega_hs.split(",")] if args.omega_hs else list(W0)[::args.stride]
        only = {c: stable_frames_from_summary(args.only_stable_from, c) for c in ids} if args.only_stable_from else None
        scan_frames(ids, al, ws, _numerics(args), args.out, args.nproc, coherence=not args.no_coherence, resolve=args.resolve,
                    only=only, overwrite=args.overwrite, per_case_tmax=args.tmax is None)
    elif args.cmd == "summary":
        from .scan import summarise
        summarise(args.dir)
    elif args.cmd == "tempo":
        from .tempo import run_tempo
        p = get_case(args.case)
        df = run_tempo(p, tmax=args.tmax, dt=args.dt, epsrel=args.eps)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False); print("written", args.out, "for", case_id(p))


if __name__ == "__main__":
    main()
