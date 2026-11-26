#!/usr/bin/env bash
# TEMPO time-step / truncation convergence for all cases, then the analysis against the production files.
# Usage: bash scripts/goa/run_tempo_dt.sh [NPROC=4]          (keep NPROC <= free RAM in GB / 6)   -> work/tempo_conv/
source "$(dirname "$0")/common.sh"; need numpy scipy pandas oqupy
NP=${1:-4}
mkdir -p "$W"; "$PY" -m goa_pqme.studies.tempo_conv --nproc "$NP" --out "$W/tempo_conv" 2>&1 | tee -a "$W/tempo_conv.log"
"$PY" -m goa_pqme.studies.tempo_conv --analyze --out "$W/tempo_conv"
