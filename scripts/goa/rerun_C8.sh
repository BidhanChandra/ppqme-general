#!/usr/bin/env bash
# Recompute case C8 (Gamma = 1, fast dynamics) in work/scan with panel width 2 (populations and coherence); the
# summary is rebuilt over ALL frames automatically.   Usage: bash scripts/goa/rerun_C8.sh [NPROC=8] [SCAN_DIR=work/scan]
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
NP=${1:-8}; OUT=${2:-$W/scan}; mkdir -p "$OUT"
"$PY" -m goa_pqme scan --cases G1_OD0_Binf_y0p75 --nproc "$NP" --numerics converged --panel-h 2 --overwrite --out "$OUT" 2>&1 | tee -a "$OUT/run_C8.log"
