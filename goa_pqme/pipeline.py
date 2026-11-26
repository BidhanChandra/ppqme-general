"""One frame end to end:  model + (alpha, omega_h) + numerics  ->  dP(t) for TCL2 / +K3 / +K4 / TCL234.

    frame  = make_frame(params, alpha, omega_h)                 # f, B, lambda_p, H~_S, E, U, S_a
    kn     = K_nodes(frame, tmax, node_dt, nq, panel_h)         # K2, K3, K4 at sparse nodes (interaction picture)
    tcl2   = TCL2(frame, t)                                     # C_ab, rate tensors, L(t) on the fine grid
    L_var  = L + spline(P K3 P^-1) [+ spline(P K4 P^-1)]
    dP     = population_difference(propagate(L_var, t, rho0))
"""
from __future__ import annotations

import time

import numpy as np

from .config import CONVERGED, ModelParams, Numerics
from .dynamics import (FrameResult, interpolate_generator, population_difference, propagate, q1, rho0_donor,
                       to_site, variant_generators)
from .frame import make_frame
from .tcl2 import TCL2
from .tcl_higher import K_nodes, generator_ratio


def channel_variants(L, kn, E, t) -> dict:
    """Extra variants from the nX-resolved generators: K3_x0..3, K4_x0..4, hop, lin, mix."""
    S3 = [interpolate_generator(kn.t, kn.K[3][c], E, t) for c in range(4)]
    S4 = [interpolate_generator(kn.t, kn.K[4][c], E, t) for c in range(5)]
    out = {"K3_x%d" % c: L + S3[c] for c in range(4)}
    out.update({"K4_x%d" % c: L + S4[c] for c in range(5)})
    out["hop"] = L + S3[0] + S4[0]
    out["lin"] = L + S3[3] + S4[4]
    out["mix"] = L + S3[1] + S3[2] + S4[1] + S4[2] + S4[3]
    return out


def run_frame(params: ModelParams, alpha: float, omega_h: float, numerics: Numerics = CONVERGED,
              coherence: bool = False, resolve: bool = False, keep_rho=()) -> FrameResult:
    """coherence: also B rho~_DA and B rho~_DA + Q1 for TCL2, TCL234;  resolve: nX classes + channel variants;
    keep_rho: variants whose site density matrices rho(t) (nt,2,2) are returned."""
    t0 = time.time()
    frame = make_frame(params, alpha, omega_h)
    kn = K_nodes(frame, numerics.tmax, numerics.node_dt, numerics.nq, numerics.panel_h, resolve=resolve)
    t = np.arange(0.0, kn.t[-1] + 1e-9, numerics.dt)
    tcl2 = TCL2(frame, t)
    K3s = interpolate_generator(kn.t, kn.total(3), frame.E, t)
    K4s = interpolate_generator(kn.t, kn.total(4), frame.E, t)
    gens = variant_generators(tcl2.L, K3s, K4s)
    if resolve:
        gens.update(channel_variants(tcl2.L, kn, frame.E, t))
    rho0 = rho0_donor(frame)
    dP, coh, rhos = {}, {}, {}
    for name, Lv in gens.items():
        with np.errstate(all="ignore"):
            rho = propagate(Lv, t, rho0)
            dP[name] = population_difference(frame, rho)
            if coherence and name in ("TCL2", "TCL234"):
                rd = to_site(frame, rho)[:, 0, 1]
                coh[name + "_B"] = frame.B * rd
                coh[name + "_Q1"] = frame.B * rd + q1(frame, t, tcl2.C, tcl2.Cm, rho)
            if name in keep_rho:
                rhos[name] = to_site(frame, rho)
    return FrameResult(alpha=float(alpha), omega_h=float(omega_h), t=t, dP=dP, r3=generator_ratio(kn, 3),
                       r4=generator_ratio(kn, 4), B=frame.B, lambda_p=frame.lambda_p, E=frame.E, knodes=kn,
                       coherence=coh, seconds=time.time() - t0, U=frame.U, rho=rhos)
