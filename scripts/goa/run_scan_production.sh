#!/usr/bin/env bash
# Frame scan with the first production numerics (node spacing 3, nq 4, panel 6) AND its frame grid (--grid legacy: alpha 5.5,
# omega_h 0.1/0.2/0.3 ...), generators stored: reproduces results_full.jsonl / the AL cache.  Use --grid paper via EXTRA args.
# Usage: bash scripts/goa/run_scan_production.sh [NPROC=2] [CASES=all]      -> work/scan_production/
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
NP=${1:-2}; CASES=${2:-all}; shift $(( $# > 2 ? 2 : $# ))
log $W/scan_production/run.log
"$PY" -m goa_pqme scan --cases "$CASES" --nproc "$NP" --numerics production --grid legacy --no-coherence --out $W/scan_production "$@" 2>&1 | tee -a $W/scan_production/run.log
