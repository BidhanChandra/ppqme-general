# ppqme-general

A small, transparent Python implementation of the **homogeneous finite-time TCL2 partial-polaron quantum master equation (p-PQME)** for an arbitrary finite-dimensional electronic Hamiltonian coupled to **independent harmonic baths**.

The code grew from 3-state and 5-state singlet-fission notebooks, but the solver itself is not tied to singlet fission. The number of electronic states, Hamiltonian, state labels, bath parameters, initial density matrix, and propagation settings are all inputs.

## Scope

Implemented:

- arbitrary Hermitian `N x N` diabatic/site-basis Hamiltonians;
- one independent spectral density per diabatic state;
- site-dependent bath parameters;
- partial-polaron transformation with
  `W_h(omega) = 1 - exp[-(omega/omega_h)^alpha]`;
- automatic residual diagonal and nonzero hopping channels;
- finite-time TCL2 rate integrals;
- homogeneous p-PQME propagation;
- finite-time Redfield limit through `method="redfield"` (`W_h = 0`);
- localized or general density-matrix initial conditions;
- population, trace, Hermiticity, and positivity diagnostics.

Not implemented in the default production path:

- correlated baths;
- non-Hermitian Hamiltonians;
- time-dependent Hamiltonians;
- non-TCL2 memory kernels;
- the expensive inhomogeneous initial-correlation corrections. Experimental versions of the latter remain under `advanced/`.

## Conventions

Internally, `hbar = 1`. Energies and frequencies are in eV and internal time is in eV^-1. User-facing propagation times are in fs, using

`hbar = 0.6582119569 eV fs`.

The Hamiltonian and the initial density matrix must be expressed in the same diabatic/site basis.

## Install

```bash
git clone <your-repository-url>
cd ppqme-general
python -m pip install -e .
pytest -q
```

## Fastest way to run a new model

Copy an example JSON file and edit the Hamiltonian/bath parameters:

```bash
python scripts/run_general.py examples/three_state_sf.json --output-dir results/three_state
python scripts/run_general.py examples/five_state_nakano.json --output-dir results/five_state
```

The output directory contains:

- `populations.dat`
- `density_matrices.npy`
- `diagnostics.txt`
- `populations.png`

## Generic JSON input

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

For different independent baths on different states, replace `bath.spectral_density` by:

```json
"sites": [
  {"type": "debye", "reorganization_energy": 0.05, "cutoff": 0.18},
  {"type": "debye", "reorganization_energy": 0.03, "cutoff": 0.12},
  {"type": "debye", "reorganization_energy": 0.07, "cutoff": 0.22}
]
```

## Python API

```python
import numpy as np
from ppqme import (
    DebyeSpectralDensity,
    localized_initial_state,
    run_simulation,
)

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

## Code organization

```text
ppqme/
  bath.py          independent-bath correlation functions
  transform.py     partial-polaron weighting function
  model.py         transformed Hamiltonian + automatic residual channels
  tcl2.py          finite-time rates, generator, propagation
  core.py          basis transforms and superoperator utilities
  simulation.py    high-level arbitrary-N API
  spectral.py      reusable spectral-density constructors
  config.py        JSON input loader
  observables.py   diagnostics

models/
  singlet_fission.py   retained 3-state benchmark model

examples/
  three_state_sf.json
  five_state_nakano.json

advanced/
  inhomogeneous_first_order.py
  inhomogeneous_second_order.py

tests/
scripts/
```

## Mathematical structure

For a site-basis Hamiltonian

`H_S = sum_j E_j |j><j| + sum_(j != k) J_jk |j><k|`,

the partial-polaron transformation renormalizes site energies and hoppings. The implementation automatically constructs the residual system operators from all diagonal channels `(j,j)` and all nonzero ordered hopping channels `(j,k)`.

For independent baths, the residual correlation tensor is assembled from the site-resolved `phi_j(t)`, mixed term, and residual linear correlation. The transformed Hamiltonian is diagonalized, the finite-time TCL2 integrals are evaluated in its energy eigenbasis, and the full density matrix is propagated before being transformed back to the original site basis.

## Numerical checks before using a result scientifically

Converge at least:

1. `bath_omega_max`;
2. `frequency_points`;
3. `rate_dt_fs`;
4. `prop_dt_fs`;
5. the partial-polaron crossover `omega_h`.

Also inspect `diagnostics.txt`. TCL2/p-PQME is not guaranteed to preserve complete positivity, so a substantially negative minimum density-matrix eigenvalue is a warning that the approximation/parameters should be examined.

## Benchmarks

The repository retains the original singlet-fission benchmark utilities and tests. `examples/five_state_nakano.json` encodes the 5-state Hamiltonian used in the accompanying validation notebook; it is an example input, not a hard-coded assumption of the solver.

## Citation

If this code is used for publication, cite the p-PQME theoretical source(s) appropriate to the implementation and the specific model/parameter source used in your calculation. Add the exact bibliographic entries here before making the repository public.
