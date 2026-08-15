"""JSON configuration loader for arbitrary N-state calculations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .spectral import from_dict
from .simulation import localized_initial_state, run_simulation


def _complex_matrix(obj):
    """Accept a real matrix, or entries as [real, imag]."""
    arr = np.asarray(obj)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
        return arr.astype(complex)
    rows = []
    for row in obj:
        rows.append([
            complex(x[0], x[1]) if isinstance(x, list) and len(x) == 2 else complex(x)
            for x in row
        ])
    return np.asarray(rows, dtype=complex)


def load_config(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def build_from_config(cfg):
    H = _complex_matrix(cfg["hamiltonian_ev"])
    n = H.shape[0]
    labels = cfg.get("state_labels", [f"state_{i}" for i in range(n)])
    if len(labels) != n:
        raise ValueError("state_labels length must equal Hamiltonian dimension")

    bath_cfg = cfg["bath"]
    if "sites" in bath_cfg:
        specs = bath_cfg["sites"]
        if len(specs) != n:
            raise ValueError("bath.sites must contain one spectral density per state")
    else:
        specs = [bath_cfg["spectral_density"]] * n
    spectral_densities = [from_dict(s) for s in specs]

    init = cfg.get("initial_state", {"type": "localized", "state": 0})
    itype = init.get("type", "localized")
    if itype == "localized":
        rho0 = localized_initial_state(n, int(init["state"]))
    elif itype == "density_matrix":
        rho0 = _complex_matrix(init["matrix"])
    else:
        raise ValueError("initial_state.type must be 'localized' or 'density_matrix'")

    sim = dict(cfg.get("simulation", {}))
    result = run_simulation(
        H, spectral_densities, rho0,
        temperature_kelvin=float(bath_cfg.get("temperature_kelvin", 300.0)),
        **sim,
    )
    return labels, result
