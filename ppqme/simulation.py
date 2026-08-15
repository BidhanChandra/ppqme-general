"""High-level, model-agnostic p-PQME simulation interface."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .bath import BathNumerics
from .core import transform_state_to_energy_basis, transform_state_from_energy_basis
from .model import build_model
from .observables import populations, minimum_eigenvalue
from .tcl2 import TimeLocalRates, cumulative_rates, build_liouvillians, propagate
from .transform import PartialPolaronTransform

HBAR_EV_FS = 0.6582119569
KB_EV_K = 8.617333262145e-5


@dataclass
class SimulationResult:
    times_fs: np.ndarray
    states_site: np.ndarray
    populations: np.ndarray
    model: object
    minimum_density_eigenvalue: float

    @property
    def trace_error(self) -> float:
        tr = np.trace(self.states_site, axis1=1, axis2=2)
        return float(np.max(np.abs(tr - 1.0)))

    @property
    def hermiticity_error(self) -> float:
        return float(max(np.linalg.norm(r-r.conj().T) for r in self.states_site))


def beta_from_temperature(temperature_kelvin: float) -> float:
    T = float(temperature_kelvin)
    if T < 0:
        raise ValueError("temperature must be >= 0 K")
    return np.inf if T == 0 else 1.0 / (KB_EV_K * T)


def localized_initial_state(n: int, state: int) -> np.ndarray:
    if not (0 <= state < n):
        raise ValueError("initial state index out of range")
    rho = np.zeros((n, n), dtype=complex)
    rho[state, state] = 1.0
    return rho


def validate_density_matrix(rho, n: int, atol: float = 1e-10) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (n, n):
        raise ValueError(f"rho0 must have shape {(n,n)}")
    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise ValueError("rho0 must be Hermitian")
    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        raise ValueError("rho0 must have unit trace")
    if np.min(np.linalg.eigvalsh(0.5*(rho+rho.conj().T))) < -atol:
        raise ValueError("rho0 must be positive semidefinite")
    return rho


def run_simulation(
    hamiltonian,
    spectral_densities,
    rho0,
    *,
    temperature_kelvin=300.0,
    method="ppqme",
    omega_h=0.31,
    alpha=2.0,
    bath_omega_max=2.0,
    frequency_points=12000,
    rate_dt_fs=0.1,
    prop_dt_fs=0.5,
    final_time_fs=200.0,
    channel_threshold=1e-14,
):
    """Run homogeneous finite-time TCL2 p-PQME for an arbitrary N-state model.

    Parameters
    ----------
    hamiltonian
        NxN Hermitian diabatic/site-basis Hamiltonian in eV.
    spectral_densities
        Sequence of N independent-bath spectral density callables J_j(omega).
    rho0
        NxN initial density matrix in the same basis as ``hamiltonian``.
    method
        ``"ppqme"`` for the partial-polaron equation or ``"redfield"`` for
        its W_h=0 finite-time Redfield limit.
    """
    H = np.asarray(hamiltonian, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("hamiltonian must be square")
    n = H.shape[0]
    if len(spectral_densities) != n:
        raise ValueError("provide exactly one independent spectral density per state")
    rho0 = validate_density_matrix(rho0, n)

    if final_time_fs <= 0 or rate_dt_fs <= 0 or prop_dt_fs <= 0:
        raise ValueError("time parameters must be positive")
    if prop_dt_fs < rate_dt_fs:
        raise ValueError("prop_dt_fs must be >= rate_dt_fs")

    kind = "partial" if method.lower() == "ppqme" else "none" if method.lower() == "redfield" else None
    if kind is None:
        raise ValueError("method must be 'ppqme' or 'redfield'")

    transform = PartialPolaronTransform(
        omega_h=float(omega_h),
        alpha=float(alpha),
        kind=kind,
        frequency_points=int(frequency_points),
        omega_max=float(bath_omega_max),
    )
    tmax = float(final_time_fs) / HBAR_EV_FS
    numerics = BathNumerics(
        beta=beta_from_temperature(temperature_kelvin),
        rate_tmax=tmax,
        rate_dt=float(rate_dt_fs) / HBAR_EV_FS,
    )
    model = build_model(
        H, spectral_densities, numerics, transform,
        channel_threshold=channel_threshold,
    )

    fine = cumulative_rates(
        model.energies,
        model.metadata["rate_times"],
        model.metadata["correlation_tensor"],
        model.metadata["correlation_tensor_negative"],
    )
    stride = max(1, int(round(float(prop_dt_fs) / float(rate_dt_fs))))
    idx = np.arange(0, len(fine.times), stride)
    if idx[-1] != len(fine.times)-1:
        idx = np.append(idx, len(fine.times)-1)
    rates = TimeLocalRates(fine.times[idx], fine.W[idx], fine.W_backward[idx])

    L = build_liouvillians(model, rates)
    rho0_energy = transform_state_to_energy_basis(rho0, model.eigenvectors)
    states_energy = propagate(rates.times, L, rho0_energy)
    states_site = np.asarray([
        transform_state_from_energy_basis(r, model.eigenvectors)
        for r in states_energy
    ])
    pops = populations(states_site)
    return SimulationResult(
        times_fs=rates.times * HBAR_EV_FS,
        states_site=states_site,
        populations=pops,
        model=model,
        minimum_density_eigenvalue=minimum_eigenvalue(states_site),
    )
