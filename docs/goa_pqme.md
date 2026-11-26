# goa_pqme: partial-polaron TCL2 / TCL3 / TCL4 for the GOA donor–acceptor model

`goa_pqme` is a second package in this repository, next to the general N-state TCL2 solver `ppqme`.
It treats the two-state GOA (Garg–Onuchic–Ambegaokar) model: one primary mode Ω damped by an Ohmic secondary bath,
discretised into N = 50 normal modes. For every partial-polaron frame
$f(\omega)=1-e^{-(\omega/\omega_h)^\alpha}$ it computes

* the exact residual two-time correlation tensor and the **TCL2** generator,
* the **ordered-cumulant TCL3 and TCL4** generators $\mathcal K_3,\mathcal K_4$ from exact multi-time Gaussian moments
  of the residual (non-Gaussian) polaron bath operators, optionally split by the number of linear vertices,
* populations for TCL2, TCL2+K3, TCL2+K4 and TCL234, and the lab-frame coherence $B\tilde\rho_{DA}+Q_1$,
* frame scans over $(\alpha,\omega_h)$, comparison with numerically exact **TEMPO** (OQuPy) dynamics on the same
  discrete bath, ensemble statistics (oracle / typical / reference-free median), and **active learning** of $\mathcal K_3+\mathcal K_4$.

The full derivation, from the bath correlation function to $\mathcal K_2$, $\mathcal K_3$, $\mathcal K_4$ and the ratio $r_4$,
with the mathematics above every code cell, is in `notebooks/PPQME_Code_Walkthrough.ipynb`.

## Layout
```
goa_pqme/
  config.py      ModelParams, Numerics (PRODUCTION 3/4/6, CONVERGED 1.5/4/3), frame grids, paths, channels
  cases.py       data/cases.csv, case id <-> tag, the 20 paper cases, TEMPO reference files
  bath.py        secondary modes -> normal modes (omega_q, R_q, E_r), lab correlation C(tau)
  frame.py       filter f_q, B, lambda_p, H~_S, E, U, S_a; phi(t), M(t), C_X(t)
  tcl2.py        C_ab(t) -> rate tensors -> TCL2 generator L2(t)
  moments.py     exact multi-time Gaussian moments of the residual bath operators
  tcl_higher.py  Liouville moments M_n -> K2, K3, K4 (optionally split by nX), ratios r3, r4
  liouville.py   vec / superoperator / picture conventions
  dynamics.py    rotate + spline K_n, propagate, P_D - P_A, coherence B rho~_DA + Q1
  metrics.py     half-RMSE, stability screen, frame tables, oracle / typical / RF
  pipeline.py    run_frame(): one frame end to end
  io.py, scan.py one npz format, resumable parallel scans, run_grid
  tempo.py       OQuPy TEMPO reference
  studies/       ladder (numerical convergence), bias (bias scan + lab control), tempo_conv, checks
  analysis/      build (figures folder), paper_numbers, ensembles, lg_compare, positivity, frame_pattern,
                 mechanisms, ensemble_vs_single, channels, frame_independence
  al/            data, surrogate (PCA + GP, acquisition), study, pop_eval, transfer
scripts/goa/     bash entry points
tests/test_goa_pqme.py
notebooks/       PPQME_Code_Walkthrough.ipynb, GOA_TEMPO_TCL234_workflow.ipynb
data/cases.csv   parameters of the GOA cases
```

## Quick start
```bash
python -m pip install -e ".[tempo,al,test]"       # oqupy and scikit-learn are optional
python -m goa_pqme info  G0p2_OD0_Binf_y0p75 --alpha 2 --omega-h 0.3      # Hamiltonian, B, lambda_p, H~_S
python -m goa_pqme frame G0p2_OD0_B5_y0p75 2 0.29                           # one frame: r3, r4, max|dP| (errors if a TEMPO reference exists)
python tests/test_goa_pqme.py
```
```python
import goa_pqme as gp
p = gp.ModelParams(Gamma=0.2, omega_DA=0.0, beta=5.0, y0=0.75)
res = gp.run_frame(p, alpha=2.0, omega_h=0.29, numerics=gp.CONVERGED, coherence=True)
res.dP["TCL2"], res.dP["TCL234"], res.r4, res.coherence["TCL234_Q1"]
```
Units: $\hbar=k_B=1$, energies in units of the secondary-bath cutoff $\omega_c$ ($\Omega=0.5$), times in $1/\omega_c$.
This is unlike `ppqme`, which works in eV and fs. Every module has `--help`, e.g. `python -m goa_pqme.analysis.paper_numbers --help`.

## Scripts (`scripts/goa/`; outputs under `work/` unless `GOA_WORK` is set; interpreter via `PYTHON=...`)
| script | what |
|---|---|
| `run_tests.sh [--fast]` | regression and invariant tests |
| `run_tempo.sh CASE` | TEMPO reference of one case (production settings) -> `notebooks/csv/` |
| `run_scan.sh [NPROC] [CASES]` | all frames, converged numerics, populations + coherence -> `work/scan/` |
| `rerun_C8.sh [NPROC]` | case C8 (Γ = 1) with panel width 2 |
| `run_analysis.sh [SCAN] [FIGS] [TEMPO_DIR]` | every paper analysis from a scan -> figures folder |
| `run_channels.sh [NPROC] [STRIDE]` | channel-resolved $K_n^{(n_X)}$ scan + analysis |
| `run_convergence.sh [NPROC]` | numerical-convergence ladder (node spacing, quadrature order, panel width) |
| `run_bias.sh tempo\|tcl\|analyze\|lab` | bias scan and lab-frame control |
| `run_tempo_dt.sh [NPROC]` | TEMPO time-step convergence |
| `run_checks.sh [NPROC]` | trace / Hermiticity of the propagated state |
| `run_scan_production.sh [NPROC]` | production numerics + generators stored (input of active learning) |
| `run_al.sh [NPROC] [SCAN]` | active learning: cache, learning curves, population check |

Typical order: tests → TEMPO references → `run_scan` (+ `rerun_C8`) → `run_analysis` → supporting studies →
`run_scan_production` → `run_al`.

## Data
Only `data/cases.csv` is included. The TEMPO references (`data/trajectories/<case>.csv` and
`notebooks/csv/<tag>_tempo_T24_dt0.1_eps*.csv`) and all generated results will be added separately. Until then the
tests that need them are skipped, and everything can be regenerated with the scripts above.

## Validation
* The TCL2 generator equals the interaction-picture $\mathcal K_2=\int_0^t\mathcal M_2$ from the general moment engine
  (to the trapezoid accuracy of the rate integrals). Every $\mathcal K_n$ is trace preserving, and the two-point moments
  reproduce the closed-form correlation tensor.
* The package was checked against the original research code on identical inputs: generators, dynamics and coherences
  agree to $10^{-10}$ or better, and every analysis produced byte-identical tables.
