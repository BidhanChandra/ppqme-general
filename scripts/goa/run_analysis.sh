#!/usr/bin/env bash
# All paper analyses from a frame scan (no new dynamics, except mechanisms --generators).
# Usage: bash scripts/goa/run_analysis.sh [SCAN=work/scan] [FIGS=goa_figs_v2] [TEMPO_DIR=notebooks/csv] [COH=SCAN]
#   older-format results:  bash scripts/goa/run_analysis.sh work/scan_new goa_figs_v2 notebooks/csv work/scan_coh
# Writes FIGS/csv/*_frame_errors.csv, *_coherence_errors.csv, FIGS/curves, FIGS/*.csv tables, FIGS/figs/*, FIGS/positivity/*.
source "$(dirname "$0")/common.sh"; need numpy scipy pandas matplotlib
SCAN=${1:-$W/scan}; FIGS=${2:-goa_figs_v2}; TEMPO=${3:-notebooks/csv}; COH=${4:-$SCAN}
A="$PY -m goa_pqme.analysis"
$A.build frames    --scan "$SCAN" --tempo-dir "$TEMPO" --out "$FIGS"
$A.build coherence --scan "$SCAN" --coh-scan "$COH" --out "$FIGS"
$A.paper_numbers --data "$FIGS"            | tee "$FIGS/paper_numbers.txt"
$A.ensembles threshold --data "$FIGS"
$A.ensembles checks --data "$FIGS"         | tee "$FIGS/ensemble_checks.txt"
[ -f "$FIGS/lg_data/digitised_laigeva_curves.csv" ] && $A.lg_compare --data "$FIGS" || echo "skip lg_compare (no $FIGS/lg_data)"
$A.frame_pattern --data "$FIGS"
$A.frame_pattern --figure --data "$FIGS" --out "$FIGS/figs"
$A.mechanisms --data "$FIGS"
$A.ensemble_vs_single --data "$FIGS"
for w in audit checked sweep; do $A.positivity $w --scan "$SCAN" --coh "$COH" --tempo-dir "$TEMPO" --out "$FIGS/positivity"; done
echo "done; open $FIGS/GOA_figures.ipynb for the figures"
