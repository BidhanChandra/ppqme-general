r"""Active-learning data set from a frame scan that stored the generators (any new goa_pqme scan; use PRODUCTION numerics
for the published setting: nodes every 3 -> 8 non-zero nodes -> 8 x 16 x 2 = 256 numbers per frame).

Per frame:  dK_site(t_k) = W [P(t_k) (K3 + K4)(t_k) P(t_k)^{-1}] W^{-1},   W = U^* kron U    (Schroedinger picture, site basis),
            Y = [Re, Im] of dK_site at the non-zero nodes, flattened;  X = (alpha, omega_h);  F = (B, lambda_p, rms ||K2||);
            lab = (err TCL2, err TCL234, stable TCL234, max|dP_TCL234|)  (errors from the scan's summary.csv; nan if absent).

  python -m goa_pqme.al.data --scan work/scan_production [--out work/al/cache.npy] [--exclude-prefix G1_]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from ..config import WORK
from ..io import frame_K_total, iter_frames, load_frame
from ..liouville import energy_to_site, to_schrodinger
from ..tcl_higher import rms_norm

CACHE = os.path.join(WORK, "al", "cache.npy")


def frame_features(z: dict):
    """(Y row, F row, t_nodes) from a loaded frame with generators, E and U."""
    t = z["t_nodes"]; E = z["E"]; U = z["U"]
    dK = np.zeros((len(t), 4, 4), complex)
    dK[1:] = to_schrodinger(frame_K_total(z, 3)[1:] + frame_K_total(z, 4)[1:], E, t[1:])
    v = energy_to_site(dK, U)[1:].reshape(-1)
    return np.concatenate([v.real, v.imag]), (float(z["B"]), float(z["lambda_p"]), rms_norm(frame_K_total(z, 2))), t


def build_cache(scan: str, out: str = CACHE, exclude_prefix: str | None = "G1_") -> dict:
    labels = {}
    sp = os.path.join(scan, "summary.csv")
    if os.path.exists(sp):
        with open(sp) as fh:
            for r in csv.DictReader(fh):
                f = lambda k: float(r[k]) if r.get(k) not in (None, "") else np.nan
                labels[(r["case"], float(r["alpha"]), float(r["omega_h"]))] = (f("err_TCL2"), f("err_TCL234"), float(r.get("stable_TCL234") == "True"), f("maxabs_TCL234"))
    cache = {}
    for path, cid, a, w in iter_frames(scan):
        if exclude_prefix and cid.startswith(exclude_prefix):
            continue
        z = load_frame(path)
        if not z["K"] or "U" not in z:
            continue
        y, F, t = frame_features(z)
        c = cache.setdefault(cid, dict(Y=[], X=[], F=[], lab=[], t_nodes=t))
        c["Y"].append(y); c["X"].append((a, w)); c["F"].append(F); c["lab"].append(labels.get((cid, a, w), (np.nan,) * 4))
    out_ = {}
    for k, v in cache.items():                       # frames in numerical (alpha, omega_h) order: acquisition seeds index into it
        o = sorted(range(len(v["X"])), key=lambda i: tuple(v["X"][i]))
        out_[k] = dict(Y=np.array(v["Y"])[o], X=np.array(v["X"])[o], F=np.array(v["F"])[o], lab=np.array(v["lab"], float)[o], t_nodes=v["t_nodes"])
    cache = out_
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, cache, allow_pickle=True)
    print("cache:", out, {k: v["Y"].shape for k, v in cache.items()})
    return cache


def load_cache(path: str = CACHE) -> dict:
    return np.load(path, allow_pickle=True).item()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True); ap.add_argument("--out", default=CACHE); ap.add_argument("--exclude-prefix", default="G1_")
    a = ap.parse_args(argv)
    build_cache(a.scan, a.out, a.exclude_prefix or None)


if __name__ == "__main__":
    main()
