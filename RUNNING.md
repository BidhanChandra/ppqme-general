# Running ppqme-general

This guide explains how to install, test, and run the generalized N-state homogeneous finite-time TCL2 partial-polaron quantum master equation (p-PQME) code.

## 1. Requirements

- Python 3.10 or newer
- `pip`
- Git (only needed for cloning/version control)

The Python dependencies are installed automatically from `pyproject.toml`:

- NumPy
- SciPy
- Matplotlib

## 2. Create a clean Python environment

From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The `-e` option installs the package in editable mode, so changes to the source code are immediately available without reinstalling it.

## 3. Test the installation

Run:

```bash
pytest -q
```

All tests should pass before using the code for production calculations.

If `pytest` is not installed in your environment, install it with:

```bash
python -m pip install pytest
```

## 4. Run the smallest example first

```bash
python scripts/run_general.py examples/quickstart_two_state.json \
  --output-dir results/quickstart
```

This verifies the complete JSON -> solver -> output workflow.

## 5. Run the supplied p-PQME examples

Three-state singlet-fission model:

```bash
python scripts/run_general.py examples/three_state_sf.json \
  --output-dir results/three_state_sf
```

Five-state Nakano validation model:

```bash
python scripts/run_general.py examples/five_state_nakano.json \
  --output-dir results/five_state_nakano
```

## 6. Output files

Each calculation writes four files to the requested output directory:

- `populations.dat` - time in fs followed by the population of each electronic state
- `density_matrices.npy` - complete propagated density matrices as a NumPy array
- `diagnostics.txt` - trace, Hermiticity, positivity diagnostics, and final populations
- `populations.png` - population dynamics plot

Always inspect `diagnostics.txt` before interpreting a trajectory scientifically.

## 7. Make a new model

The easiest approach is to copy an existing JSON input:

```bash
cp examples/quickstart_two_state.json examples/my_model.json
```

Then edit `examples/my_model.json`.

A minimal three-state input has the form:

```json
{
  "state_labels": ["A", "B", "C"],
  "hamiltonian_ev": [
    [0.00, 0.02, 0.00],
    [0.02, 0.10, 0.03],
    [0.00, 0.03, 0.20]
  ],
  "initial_state": {
    "type": "localized",
    "state": 0
  },
  "bath": {
    "temperature_kelvin": 300.0,
    "spectral_density": {
      "type": "debye",
      "reorganization_energy": 0.05,
      "cutoff": 0.18
    }
  },
  "simulation": {
    "method": "ppqme",
    "omega_h": 0.10,
    "alpha": 2.0,
    "bath_omega_max": 2.0,
    "frequency_points": 12000,
    "rate_dt_fs": 0.1,
    "prop_dt_fs": 0.5,
    "final_time_fs": 200.0
  }
}
```

The Hamiltonian can be any Hermitian N x N matrix. Its dimensions must match the number of entries in `state_labels`.

## 8. Independent site-dependent baths

If each diabatic state has a different independent bath, replace the single `spectral_density` entry with `sites`:

```json
"bath": {
  "temperature_kelvin": 300.0,
  "sites": [
    {"type": "debye", "reorganization_energy": 0.05, "cutoff": 0.18},
    {"type": "debye", "reorganization_energy": 0.03, "cutoff": 0.12},
    {"type": "debye", "reorganization_energy": 0.07, "cutoff": 0.22}
  ]
}
```

There must be one bath specification per electronic state.

## 9. Redfield comparison

To use the finite-time Redfield limit with the same model, change:

```json
"method": "ppqme"
```

to:

```json
"method": "redfield"
```

The same runner and output format are then used.

## 10. Initial conditions

For a localized initial state:

```json
"initial_state": {
  "type": "localized",
  "state": 0
}
```

For an explicitly specified density matrix:

```json
"initial_state": {
  "type": "density_matrix",
  "matrix": [
    [0.5, 0.5],
    [0.5, 0.5]
  ]
}
```

The density matrix is checked for Hermiticity, unit trace, and positive semidefiniteness.

## 11. Numerical convergence

Before treating a result as converged, vary at least:

1. `bath_omega_max`
2. `frequency_points`
3. `rate_dt_fs`
4. `prop_dt_fs`
5. `omega_h`

TCL2/p-PQME dynamics is not guaranteed to preserve complete positivity. A substantially negative value of `minimum_density_eigenvalue` in `diagnostics.txt` should therefore be treated as a warning.

## 12. Python API

The solver can also be called directly from Python:

```python
import numpy as np
from ppqme import DebyeSpectralDensity, localized_initial_state, run_simulation

H = np.array([
    [0.00, 0.02],
    [0.02, 0.10],
])

baths = [
    DebyeSpectralDensity(0.05, 0.18),
    DebyeSpectralDensity(0.05, 0.18),
]

result = run_simulation(
    H,
    baths,
    localized_initial_state(2, 0),
    temperature_kelvin=300.0,
    method="ppqme",
    omega_h=0.10,
    final_time_fs=200.0,
)

print(result.populations)
print(result.trace_error)
```

## 13. Typical workflow

For a new scientific calculation, a sensible sequence is:

```bash
# activate the environment
source .venv/bin/activate

# verify the installation
pytest -q

# copy an input
cp examples/quickstart_two_state.json examples/my_model.json

# edit my_model.json, then run it
python scripts/run_general.py examples/my_model.json \
  --output-dir results/my_model

# inspect diagnostics and populations
cat results/my_model/diagnostics.txt
```

Do not commit the `results/` directory unless you intentionally want calculated results under version control; it is excluded by the supplied `.gitignore`.
