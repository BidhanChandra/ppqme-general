"""Parameters, numerical settings and paths -- the ONE place where defaults live.

Units: hbar = k_B = 1, energies in units of the secondary-bath cutoff omega_c (default 1),
times in units of 1/omega_c.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace

# ---------------------------------------------------------------------------
# Paths (no directories are created on import)
# ---------------------------------------------------------------------------
ROOT = os.environ.get("GOA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
WORK = os.environ.get("GOA_WORK", os.path.join(ROOT, "work"))
#: folder with the production TEMPO csv files named <tag>_tempo_T24_dt0.1_eps*.csv
TEMPO_DIR = os.environ.get("GOA_TEMPO_DIR", os.path.join(ROOT, "notebooks", "csv"))
#: folder used by the paper figures (csv/, curves/, figs/, GOA_figures.ipynb)
FIGS_DIR = os.environ.get("GOA_FIGS_DIR", os.path.join(ROOT, "goa_figs_v2"))


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    """GOA donor/acceptor model.

    H = H_S + H_B + |A><A| X,   H_S = [[omega_DA, Gamma], [Gamma, E_r]]  (basis D, A),
    H_B = sum_q (p_q^2 + w_q^2 q_q^2)/2,   X = -sum_q w_q^2 R_q q_q.
    The bath is one primary mode Omega (donor/acceptor minima at +-y0) damped by an Ohmic
    secondary bath (eta, omega_c) discretised into N-1 modes -> N normal modes.
    beta = inf means T = 0.
    """
    Gamma: float = 0.2
    omega_DA: float = 0.0
    beta: float = math.inf
    y0: float = 0.75
    Omega: float = 0.5
    N: int = 50
    omega_c: float = 1.0
    eta: float = 1.0

    @property
    def Er(self) -> float:
        """Reorganisation energy, exact closed form E_r = 2 Omega^2 y0^2."""
        return 2.0 * self.Omega ** 2 * self.y0 ** 2

    def with_(self, **kw) -> "ModelParams":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Numerics:
    """node_dt : spacing of the sparse time nodes where K3, K4 are evaluated
    nq      : Gauss-Legendre points per panel
    panel_h : maximal panel width of the composite quadrature
    dt      : propagation / TCL2 grid step
    tmax    : final time (rounded down to a multiple of node_dt)
    """
    node_dt: float = 1.5
    nq: int = 4
    panel_h: float = 3.0
    dt: float = 0.05
    tmax: float = 24.0

    def with_(self, **kw) -> "Numerics":
        return replace(self, **kw)


#: settings used for the published production scan (run_scan.sh / genf.py)
PRODUCTION = Numerics(node_dt=3.0, nq=4, panel_h=6.0, dt=0.05, tmax=24.0)
#: converged-ladder recommendation (scan/scan_frames.py defaults)
CONVERGED = Numerics(node_dt=1.5, nq=4, panel_h=3.0, dt=0.05, tmax=24.0)

#: production frame grid (10 x 12 = 120 frames)
ALPHAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.5)
OMEGAHS = (0.01, 0.02, 0.05, 0.095, 0.15, 0.29, 0.5, 0.8, 1.2, 2.0, 3.0, 4.0)

#: grid of the first production scan (run_scan.sh / scan/jobs_full.txt; source of results_full.jsonl and the AL cache)
LEGACY_ALPHAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 6.5, 8.0, 10.5)
LEGACY_OMEGAHS = (0.01, 0.05, 0.095, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0)

#: a frame is "stable" if dP is finite and max|dP| <= STABLE_MAX
STABLE_MAX = 1.6

#: the four system/bath channels |j><k| in storage order, and the active ones
CHANNELS = ((0, 0), (1, 1), (0, 1), (1, 0))
CHANNEL_NAMES = ("DD", "AA", "DA", "AD")
ACTIVE = (1, 2, 3)   # DD has no residual bath operator in the donor-reference frame
