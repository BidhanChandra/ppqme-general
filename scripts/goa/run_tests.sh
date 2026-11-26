#!/usr/bin/env bash
# Regression tests of goa_pqme against the legacy engine and the published frame.  Usage: bash scripts/goa/run_tests.sh [--fast]
source "$(dirname "$0")/common.sh"; need numpy scipy pandas
PYTHONDONTWRITEBYTECODE=1 "$PY" tests/test_goa_pqme.py "$@"
