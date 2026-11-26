"""Generator-level active-learning study: every case x strategy x seed, learning curves at the checkpoints CH.

  python -m goa_pqme.al.study [--cache work/al/cache.npy] [--out work/al/al_results.json] [--nproc 2] [--seeds 4]
Configurations: (random, F0), (maximin, F0), (gp_u, F0), (gp_u, F3), (hybrid, F0).
"""
from __future__ import annotations

import argparse
import json
import os

from .data import CACHE, load_cache
from .surrogate import run

CFG = [("random", "F0"), ("maximin", "F0"), ("gp_u", "F0"), ("gp_u", "F3"), ("hybrid", "F0")]
_CACHE = {}


def _work(j):
    path, c, s, k, seed = j
    if path not in _CACHE:
        _CACHE[path] = load_cache(path)
    return dict(case=c, strategy=s, feat=k, seed=seed, res=run(_CACHE[path][c], s, k, seed))


def main(argv=None):
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=CACHE); ap.add_argument("--out", default=None); ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--cases", default="all")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cache)), "al_results.json")
    cases = list(load_cache(a.cache)) if a.cases == "all" else a.cases.split(",")
    jobs = [(a.cache, c, s, k, seed) for c in cases for (s, k) in CFG for seed in range(a.seeds)]
    res = []
    with Pool(a.nproc) as p:
        for i, r in enumerate(p.imap_unordered(_work, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(i, len(jobs), flush=True); json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w")); print("written", out)


if __name__ == "__main__":
    main()
