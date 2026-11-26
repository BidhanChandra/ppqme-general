"""Case registry: data/cases.csv <-> ModelParams, case-id / tag conversions, TEMPO references.

Two naming conventions exist in the project and are converted here (and only here):
  case id : G0p2_OD0p28125_B5_y0p75                         (scan files, data/trajectories)
  tag     : G0.2_OD0.28125_B5_y0.75_W0.5_N50_wc1_eta1       (notebook / TEMPO csv files)
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re

from .config import DATA, ModelParams

_ID_RE = re.compile(r"^G([0-9p.]+)_OD([0-9p.]+)_B(inf|[0-9p.]+)_y([0-9p.]+)")


def _num(s: str) -> float:
    return float(s.replace("p", "."))


def _fmt(x: float) -> str:
    return ("%g" % x).replace(".", "p")


def case_id(p: ModelParams) -> str:
    """ModelParams -> 'G0p2_OD0_Binf_y0p75' (bath defaults are not encoded)."""
    b = "inf" if math.isinf(p.beta) else _fmt(p.beta)
    return "G%s_OD%s_B%s_y%s" % (_fmt(p.Gamma), _fmt(p.omega_DA), b, _fmt(p.y0))


def tag(p: ModelParams) -> str:
    """ModelParams -> notebook/TEMPO key 'G0.2_OD0_Binf_y0.75_W0.5_N50_wc1_eta1'."""
    return "G%g_OD%g_B%g_y%g_W%g_N%d_wc%g_eta%g" % (p.Gamma, p.omega_DA, p.beta, p.y0, p.Omega, p.N, p.omega_c, p.eta)


def parse_case_id(s: str, **bath) -> ModelParams:
    """'G0p2_OD0_Binf_y0p75' or a tag 'G0.2_OD0_Binf_y0.75_...' -> ModelParams (bath defaults unless given)."""
    m = _ID_RE.match(s)
    if not m:
        raise ValueError("cannot parse case id %r" % s)
    g, od, b, y = m.groups()
    return ModelParams(Gamma=_num(g), omega_DA=_num(od), beta=math.inf if b == "inf" else _num(b), y0=_num(y), **bath)


def load_cases(path: str | None = None) -> dict[str, ModelParams]:
    """Read data/cases.csv -> {case_id: ModelParams}."""
    path = path or os.path.join(DATA, "cases.csv")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            beta = math.inf if r["beta"].strip() == "inf" else float(r["beta"])
            out[r["case_id"]] = ModelParams(Gamma=float(r["Gamma"]), omega_DA=float(r["omega_DA"]), beta=beta, y0=float(r["y0"]))
    return out


def get_case(cid: str) -> ModelParams:
    cases = load_cases()
    return cases[cid] if cid in cases else parse_case_id(cid)


def reference_file(cid: str, folder: str | None = None) -> str | None:
    """TEMPO reference for a case id: '<cid>.csv' if present, else the first (sorted) '<cid>_*.csv'."""
    folder = folder or os.path.join(DATA, "trajectories")
    exact = os.path.join(folder, cid + ".csv")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(folder, cid + "_*.csv")))
    return cands[0] if cands else None


# ---------------------------------------------------------------------------------------------
# The 20 paper cases: development set C1..C14 and held-out set H1..H6 (prefixes of the tag)
# ---------------------------------------------------------------------------------------------
DEV_PREFIXES = ("G0.1_OD0_Binf", "G0.2_OD0_Binf", "G0.2_OD0.1_Binf", "G0.2_OD0.2_Binf", "G0.2_OD0.28125_Binf", "G0.2_OD0.5_Binf",
                "G0.2_OD1_Binf", "G1_OD0_Binf", "G0.2_OD0_B5_y0.75", "G0.2_OD0.28125_B5", "G0.2_OD0_B1_y0.25", "G0.2_OD0_B1_y0.75",
                "G0.2_OD0_B1_y1.5", "G0.2_OD0.28125_B1")
HELD_PREFIXES = ("G0.2_OD0_B10", "G0.2_OD0.28125_B10", "G0.2_OD0_B2", "G0.2_OD0.28125_B2", "G0.2_OD0_B0.5", "G0.2_OD0.28125_B0.5")


def stem_of(key: str) -> str:
    """tag 'G0.2_OD0_Binf_y0.75_W0.5_...' -> case id 'G0p2_OD0_Binf_y0p75'."""
    g = re.match(r"G([\d.]+)_OD([\d.]+)_B(inf|[\d.]+)_y([\d.]+)_", key).groups()
    return "G%s_OD%s_B%s_y%s" % tuple(x.replace(".", "p") for x in g)


def key_of(cid: str) -> str:
    """case id 'G0p2_OD0_Binf_y0p75' -> tag prefix 'G0.2_OD0_Binf_y0.75' (p between digits -> .)."""
    return re.sub(r"(?<=\d)p(?=\d)", ".", cid)


def match_prefix(prefix: str, names) -> str:
    """The unique name (tag or case id) that starts with a DEV/HELD prefix."""
    pp, pc = prefix + "_", prefix.replace(".", "p") + "_"
    hits = [k for k in sorted(names) if k.startswith(pp) or (k + "_").startswith(pc) or k.startswith(pc)]
    if not hits:
        raise KeyError("no case for prefix %r" % prefix)
    return hits[0]


def case_labels(names) -> dict:
    """{name: 'C1'..'C14' / 'H1'..'H6'} for the names (tags or case ids) that belong to the paper cases."""
    lab = {}
    for i, p in enumerate(DEV_PREFIXES):
        try:
            lab[match_prefix(p, names)] = "C%d" % (i + 1)
        except KeyError:
            pass
    for i, p in enumerate(HELD_PREFIXES):
        try:
            lab[match_prefix(p, names)] = "H%d" % (i + 1)
        except KeyError:
            pass
    return lab


def tempo_file(folder: str, key: str) -> str:
    """Production TEMPO csv for a tag: first sorted '<key>_tempo*' containing 'dt0.1' and '_T24_' (any T for G1_)."""
    fs = sorted(f for f in glob.glob(os.path.join(folder, key + "_tempo*")) if "dt0.1" in f)
    fs = [f for f in fs if "_T24_" in f or key.startswith("G1_")]
    if not fs:
        raise FileNotFoundError("no dt0.1 TEMPO file for %s in %s" % (key, folder))
    return fs[0]


def case_tmax(cid: str, cap: float = 24.0) -> float:
    """Time horizon of a case: last time of its TEMPO reference (capped at `cap`), or `cap` if there is none."""
    try:
        import pandas as pd
        d = pd.read_csv(reference_file(cid), usecols=lambda c: c in ("t", "t_au"))
        return float(min(cap, d.iloc[:, 0].max()))
    except Exception:
        return float(cap)


def load_reference(cid_or_path: str):
    """TEMPO reference as a DataFrame with columns t, P_D, P_A, dP, Re_rho_DA, Im_rho_DA."""
    import pandas as pd
    path = cid_or_path if os.path.exists(cid_or_path) else reference_file(cid_or_path)
    if path is None:
        raise FileNotFoundError("no TEMPO reference for %r" % cid_or_path)
    df = pd.read_csv(path)
    if "t" not in df.columns and "t_au" in df.columns:
        df = df.rename(columns={"t_au": "t"})
    return df
