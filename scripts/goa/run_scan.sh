#!/usr/bin/env bash
# Frame scan (TCL2 / +K3 / +K4 / TCL234 populations + coherence B*rho~_DA and +Q1) for all cases, converged numerics
# (node spacing 1.5, nq 4, panel 3; tmax = min(24, TEMPO horizon of the case)).  Resumable.
# Usage: bash scripts/goa/run_scan.sh [NPROC=2] [CASES=all] [EXTRA goa_pqme scan options...]
#   bash scripts/goa/run_scan.sh 8                       -> work/scan/curves/*.npz, work/scan/summary.csv
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
NP=${1:-2}; CASES=${2:-all}; shift $(( $# > 2 ? 2 : $# ))
log $W/scan/run.log
"$PY" -m goa_pqme scan --cases "$CASES" --nproc "$NP" --numerics converged --out $W/scan "$@" 2>&1 | tee -a $W/scan/run.log
