#!/usr/bin/env bash
# Channel-resolved scan: K2, K3, K4 split by the number of linear vertices nX, and propagations of TCL2 + each class
# (K3_x0..3, K4_x0..4, hop, lin, mix).  Then the channel analysis.
# Usage: bash scripts/goa/run_channels.sh [NPROC=8] [STRIDE=2] [CASES=all]   (stride 2: 5 x 6 frames per case; 1: full grid)
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
NP=${1:-8}; STRIDE=${2:-2}; CASES=${3:-all}
log $W/channels/run.log
"$PY" -m goa_pqme scan --cases "$CASES" --nproc "$NP" --numerics converged --resolve --no-coherence --stride "$STRIDE" --out $W/channels 2>&1 | tee -a $W/channels/run.log
"$PY" -m goa_pqme.analysis.channels --scan $W/channels --out $W/channels_analysis
