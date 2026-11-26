#!/usr/bin/env bash
# TEMPO reference for one case (production settings: dt 0.1, eps 1e-8 at beta=inf, 1e-9 otherwise).
# Usage: bash scripts/goa/run_tempo.sh CASE [TMAX=24] [DT=0.1]       -> notebooks/csv/<tag>_tempo_T<TMAX>_dt<DT>_eps<eps>.csv
source "$(dirname "$0")/common.sh"; need numpy scipy pandas oqupy
CASE=${1:?case id, e.g. G0p2_OD0_Binf_y0p75}; TMAX=${2:-24}; DT=${3:-0.1}
OUT=$("$PY" -c "import goa_pqme as g; from goa_pqme.tempo import production_epsrel as e; p=g.get_case('$CASE'); print('notebooks/csv/%s_tempo_T%g_dt%g_eps%g.csv' % (g.tag(p), $TMAX, $DT, e(p.beta)))")
"$PY" -m goa_pqme tempo "$CASE" --tmax "$TMAX" --dt "$DT" --out "$OUT"
