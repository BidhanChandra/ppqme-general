#!/usr/bin/env bash
# Active learning: cache from a production scan, generator-level study, population check for some cases.
# Usage: bash scripts/goa/run_al.sh [NPROC=2] [SCAN=work/scan_production] [POP_CASES=G0p2_OD0_Binf_y0p75]   (needs scikit-learn)
source "$(dirname "$0")/common.sh"; need numpy scipy pandas sklearn
NP=${1:-2}; SCAN=${2:-$W/scan_production}; POP=${3:-G0p2_OD0_Binf_y0p75}
"$PY" -m goa_pqme.al.data  --scan "$SCAN" --out $W/al/cache.npy
"$PY" -m goa_pqme.al.study --cache $W/al/cache.npy --nproc "$NP"
for c in ${POP//,/ }; do "$PY" -m goa_pqme.al.pop_eval "$c" --scan "$SCAN" --cache $W/al/cache.npy; done
