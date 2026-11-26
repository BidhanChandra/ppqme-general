#!/usr/bin/env bash
# Bias scan (Gamma 0.2, y0 0.75, omega_DA list; frames lab / mid / dressed) and lab-frame control.
# Usage: bash scripts/goa/run_bias.sh tempo   [NPROC=4] [BETAS=5]      TEMPO references (needs oqupy; ~6 GB RAM per run)
#        bash scripts/goa/run_bias.sh tcl     [NPROC=8] [BETAS=5]      channel-resolved TCL, three frames per point
#        bash scripts/goa/run_bias.sh analyze                          -> work/bias_scan/analysis/
#        bash scripts/goa/run_bias.sh lab     [NPROC=4] [BETAS=inf,10,5,2]   lab-frame control vs TEMPO (notebooks/csv)
source "$(dirname "$0")/common.sh"
MODE=${1:?mode: tempo|tcl|analyze|lab}; NP=${2:-4}; BETAS=${3:-}
case "$MODE" in
  tempo)   need numpy scipy pandas oqupy; "$PY" -m goa_pqme.studies.bias tempo --nproc "$NP" --betas "${BETAS:-5}" ;;
  tcl)     need numpy scipy pandas;       "$PY" -m goa_pqme.studies.bias tcl --nproc "$NP" --betas "${BETAS:-5}" ;;
  analyze) need numpy scipy pandas;       "$PY" -m goa_pqme.studies.bias analyze ;;
  lab)     need numpy scipy pandas;       "$PY" -m goa_pqme.studies.bias lab --nproc "$NP" --betas "${BETAS:-inf,10,5,2}" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
