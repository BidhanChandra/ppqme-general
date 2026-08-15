"""Small linear-algebra core used by the p-PQME implementation.

Conventions
-----------
* hbar = 1 internally.
* Energies/frequencies are in eV, therefore internal time is eV^-1.
* Density matrices use column-major (Fortran) vectorization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

@dataclass(frozen=True)
class ResidualChannel:
    label: tuple[int, int]
    system_operator: ComplexArray
    kappa: float = 1.0

@dataclass
class PTModel:
    h_tilde_site: ComplexArray
    energies: RealArray
    eigenvectors: ComplexArray  # U[j,p] = <j|phi_p>
    channels: list[ResidualChannel]
    beta: float
    metadata: dict[str, object]


def operator_superoperator(n: int, action: Callable[[ComplexArray], ComplexArray]) -> ComplexArray:
    out = np.empty((n*n, n*n), dtype=complex)
    for j in range(n):
        for i in range(n):
            rho = np.zeros((n,n), dtype=complex)
            rho[i,j] = 1.0
            out[:, i + j*n] = action(rho).reshape(-1, order='F')
    return out


def hamiltonian_superoperator(H: ArrayLike) -> ComplexArray:
    H = np.asarray(H, dtype=complex)
    n = H.shape[0]
    return operator_superoperator(n, lambda rho: -1j*(H@rho-rho@H))


def energy_hamiltonian_superoperator(energies: ArrayLike) -> ComplexArray:
    E = np.asarray(energies, dtype=float)
    return hamiltonian_superoperator(np.diag(E.astype(complex)))


def transform_state_to_energy_basis(rho_site: ArrayLike, U: ArrayLike) -> ComplexArray:
    rho = np.asarray(rho_site, dtype=complex)
    U = np.asarray(U, dtype=complex)
    return U.conj().T @ rho @ U


def transform_state_from_energy_basis(rho_energy: ArrayLike, U: ArrayLike) -> ComplexArray:
    rho = np.asarray(rho_energy, dtype=complex)
    U = np.asarray(U, dtype=complex)
    return U @ rho @ U.conj().T


def trace_preservation_error(L: ArrayLike) -> float:
    L = np.asarray(L, dtype=complex)
    n = int(round(np.sqrt(L.shape[0])))
    tr = np.eye(n, dtype=complex).reshape(-1, order='F')
    return float(np.max(np.abs(tr.conj() @ L)))
