#!/usr/bin/env bash
# Numerical-convergence ladder (node spacing, quadrature order, panel width) for all cases with a TEMPO reference.
# Usage: bash scripts/goa/run_convergence.sh [NPROC=2] [CASES=all] [FRAMES=alpha:omega_h,...] [TMAX=24]
# Resumable (work/ladder/runs); output work/ladder/ladder_all.csv and ladder_summary.csv.
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
NP=${1:-2}; CASES=${2:-all}; FRAMES=${3:-"0.5:0.05,0.5:0.29,1.5:0.095,2:0.5,4:0.15,8:0.02"}; TMAX=${4:-24}
log $W/ladder/run.log
"$PY" -m goa_pqme.studies.ladder --cases "$CASES" --frames "$FRAMES" --tmax "$TMAX" --nproc "$NP" --out $W/ladder 2>&1 | tee -a $W/ladder/run.log
