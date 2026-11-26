#!/usr/bin/env bash
# Trace / Hermiticity of the propagated state on a sample of frames.   Usage: bash scripts/goa/run_checks.sh [NPROC=4]
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
"$PY" -m goa_pqme.studies.checks --nproc "${1:-4}"
