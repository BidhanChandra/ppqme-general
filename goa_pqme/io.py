"""One on-disk format for a frame result, and ONE loader that also reads every legacy layout.

New format  <out>/curves/<case_id>__a<alpha>_w<omega_h>.npz :
    t, dP_<variant>, r3, r4, B, lambda_p, E, U, alpha, omega_h, seconds,
    t_nodes, K2, K3, K4 (interaction picture; (n+1, nt, 4, 4) if resolved by nX, flag 'resolved'),
    coh_<name> (complex coherence readouts), rho_<variant> (site density matrices, optional),
    meta_<param> (ModelParams), num_<setting> (Numerics), extra_<key> (free metadata).
Writes are atomic (tmp file + os.replace).

load_frame() returns a dict with the normalised keys
    t, dP{variant}, coherence{name}, rho{variant}, t_nodes, K{n}, resolved, params{}, numerics{}, + scalars,
and understands the legacy files of scan/scan_frames.py (TCL2, K3, ...), scan_coherence.py (TCL2_B, ...),
scan_channels.py / bias_scan.py (K2c, dP_*, lam_p, E0, E1, Gamma, ...), convergence_ladder.py (tK, K2, ...),
genf.py (t = nodes, K2, K3, K4, E, U, B, lp) and propS.py (t, ref, TCL2, ...).
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict

import numpy as np

from .config import ModelParams, Numerics
from .dynamics import FrameResult

VARIANTS4 = ("TCL2", "K3", "K4", "TCL234")
_PARAM_LEGACY = {"Gamma": "Gamma", "omega_DA": "omega_DA", "beta": "beta", "y0": "y0"}
_NUM_LEGACY = ("tmax", "node_dt", "nq", "panel_h")
_FNAME = re.compile(r"(.+?)__a([\d.eE+-]+)_w([\d.eE+-]+)(?:_.*)?\.npz$")


def frame_filename(out: str, cid: str, alpha: float, omega_h: float, suffix: str = "") -> str:
    return os.path.join(out, "curves", "%s__a%g_w%g%s.npz" % (cid, alpha, omega_h, suffix))


def parse_frame_filename(path: str):
    """'<cid>__a<alpha>_w<omega_h>[...].npz' -> (cid, alpha, omega_h) or None."""
    m = _FNAME.match(os.path.basename(path))
    return (m.group(1), float(m.group(2)), float(m.group(3))) if m else None


def save_frame(path: str, res: FrameResult, params: ModelParams, numerics: Numerics, keep_K: bool = True, extra: dict | None = None) -> None:
    d = dict(t=res.t, r3=res.r3, r4=res.r4, B=res.B, lambda_p=res.lambda_p, E=res.E, seconds=res.seconds,
             alpha=res.alpha, omega_h=res.omega_h)
    if res.U is not None:
        d["U"] = res.U
    d.update({"dP_" + k: v for k, v in res.dP.items()})
    d.update({"coh_" + k: v for k, v in res.coherence.items()})
    d.update({"rho_" + k: v for k, v in (res.rho or {}).items()})
    if keep_K and res.knodes is not None:
        d.update(t_nodes=res.knodes.t, resolved=res.knodes.resolved, **{"K%d" % n: K for n, K in res.knodes.K.items()})
    d.update({"meta_" + k: v for k, v in asdict(params).items()})
    d.update({"num_" + k: v for k, v in asdict(numerics).items()})
    d.update({"extra_" + k: v for k, v in (extra or {}).items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def _scalar(v):
    return v[()] if isinstance(v, np.ndarray) and v.ndim == 0 else v


def load_frame(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        raw = {k: _scalar(z[k]) for k in z.files}
    out = {"dP": {}, "coherence": {}, "rho": {}, "params": {}, "numerics": {}, "extra": {}, "K": {}, "path": path}
    rest = {}
    for k, v in raw.items():
        if k.startswith("dP_"):
            out["dP"][k[3:]] = v
        elif k.startswith("coh_"):
            out["coherence"][k[4:]] = v
        elif k.startswith("rho_"):
            out["rho"][k[4:]] = v
        elif k.startswith("meta_"):
            out["params"][k[5:]] = v
        elif k.startswith("num_"):
            out["numerics"][k[4:]] = v
        elif k.startswith("extra_"):
            out["extra"][k[6:]] = v
        else:
            rest[k] = v
    # ---- generators at nodes
    if "K2c" in rest:                                    # scan_channels / bias_scan
        out["K"] = {n: rest.pop("K%dc" % n) for n in (2, 3, 4) if "K%dc" % n in rest}
        out["resolved"] = True
    elif "K2" in rest:
        out["K"] = {n: rest.pop("K%d" % n) for n in (2, 3, 4) if "K%d" % n in rest}
        out["resolved"] = bool(rest.pop("resolved", out["K"][2].ndim == 4))
    has_curves = any(m in rest for m in VARIANTS4) or bool(out["dP"])
    if "t_nodes" in rest:
        out["t_nodes"] = rest.pop("t_nodes")
    elif "tK" in rest:                                   # convergence ladder
        out["t_nodes"] = rest.pop("tK")
    elif out["K"] and not has_curves and "t" in rest:    # genf.py: t are the nodes
        out["t_nodes"] = rest.pop("t")
    # ---- legacy curves and coherence
    for m in VARIANTS4:
        if m in rest and m not in out["dP"]:
            out["dP"][m] = rest.pop(m)
    for k in list(rest):
        if re.match(r"TCL(2|234)_(B|Q1)$", k):
            out["coherence"][k] = rest.pop(k)
    # ---- legacy scalars
    if "lam_p" in rest:
        out["lambda_p"] = float(rest.pop("lam_p"))
    if "lp" in rest:
        out["lambda_p"] = float(rest.pop("lp"))
    if "E0" in rest and "E1" in rest:
        out["E"] = np.array([float(rest.pop("E0")), float(rest.pop("E1"))])
    for k in _PARAM_LEGACY:
        if k in rest and k not in out["params"]:
            out["params"][k] = rest.pop(k)
    for k in _NUM_LEGACY:
        if k in rest and k not in out["numerics"]:
            out["numerics"][k] = rest.pop(k)
    if "sec" in rest:
        out["seconds"] = float(rest.pop("sec"))
    out.update(rest)
    parsed = parse_frame_filename(path)
    if parsed:
        out.setdefault("case", parsed[0])
        out.setdefault("alpha", parsed[1])
        out.setdefault("omega_h", parsed[2])
    return out


def frame_K_total(z: dict, n: int) -> np.ndarray:
    """Total K_n at the nodes from a loaded frame (sums the nX classes if resolved)."""
    K = z["K"][n]
    return K.sum(0) if z.get("resolved") else K


def iter_frames(folder: str):
    """Yield (path, case_id, alpha, omega_h) for every frame file in <folder>/curves (or folder itself)."""
    d = os.path.join(folder, "curves") if os.path.isdir(os.path.join(folder, "curves")) else folder
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz") and not fn.endswith(".tmp.npz"):
            p = parse_frame_filename(fn)
            if p:
                yield os.path.join(d, fn), p[0], p[1], p[2]
