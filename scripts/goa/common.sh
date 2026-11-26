# Sourced by every script in scripts/.  Runs from the package root; one BLAS thread per process.
# Choose the interpreter with PYTHON=..., e.g.  PYTHON=/path/to/env/bin/python bash scripts/goa/run_scan.sh 8
# (TEMPO needs oqupy, active learning needs scikit-learn).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export GOA_ROOT="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON:-python3}"
W="${GOA_WORK:-work}"          # output root (override with GOA_WORK)
export GOA_WORK="$W"
need() {   # need numpy scipy pandas [oqupy] [sklearn]
  "$PY" -c "import $(IFS=,; echo "$*")" 2>/dev/null || { echo "ERROR: '$PY' cannot import: $*   (set PYTHON=/path/to/python)"; exit 1; }
}
log() { mkdir -p "$(dirname "$1")"; echo "log: $1"; }
