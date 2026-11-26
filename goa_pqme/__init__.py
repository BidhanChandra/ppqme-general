"""goa_pqme -- partial-polaron TCL2/3/4 for the GOA donor/acceptor model.

Pipeline, one module per physical step:

    config      ModelParams, Numerics (PRODUCTION, CONVERGED), frame grid, channels, paths
    cases       data/cases.csv, case id <-> tag, TEMPO reference files
    bath        secondary modes -> normal modes (omega_q, R_q, E_r), lab correlation C(tau)
    frame       filter f_q, B, lambda_p, H~_S, E, U, S_a; primitive functions phi, mixed, linear
    tcl2        C_ab(t) -> rate tensors -> TCL2 generator L(t)
    moments     exact multi-time Gaussian moments of the residual bath operators
    tcl_higher  Liouville moments M_n, ordered cumulants K2, K3, K4, ratios r3, r4
    liouville   vec / superoperator / picture conventions
    dynamics    spline + rotate K_n, propagate, dP, coherence readout Q1
    metrics     half-RMSE, stability screen, oracle / typical / RF
    pipeline    run_frame(): everything above for one frame
    io, scan    on-disk format, resumable scans, run_grid;   tempo: OQuPy reference
Subpackages:
    studies     ladder, bias (+ lab control), tempo_conv, checks
    analysis    build, paper_numbers, ensembles, lg_compare, positivity, frame_pattern, mechanisms, ensemble_vs_single,
                channels, frame_independence
    al          data, surrogate, study, pop_eval, transfer
Command line:  python -m goa_pqme --help ;  python -m goa_pqme.<subpackage>.<module> --help ;  bash scripts/*.sh
"""
from .bath import BathModes, lab_correlation, normal_modes
from .cases import case_id, get_case, load_cases, load_reference, parse_case_id, tag
from .config import ALPHAS, CONVERGED, OMEGAHS, PRODUCTION, STABLE_MAX, ModelParams, Numerics
from .dynamics import FrameResult, interpolate_generator, population_difference, propagate, rho0_donor, to_site
from .frame import PolaronFrame, make_frame, polaron_filter
from .metrics import ensemble_summary, half_rmse, is_stable
from .moments import ResidualBath
from .pipeline import run_frame
from .tcl2 import TCL2, correlation_tensors, tcl2_generator
from .tcl_higher import KNodes, K_nodes, MomentEngine, generator_ratio, nested_nodes

__version__ = "1.0.0"
