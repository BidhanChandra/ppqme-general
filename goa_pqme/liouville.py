r"""Liouville-space conventions (used by every other module).

Column-major vectorisation:  vec(rho)_{i + 2j} = rho_ij,  i.e.  (rho_00, rho_10, rho_01, rho_11),
and  vec(X rho Y) = (Y^T kron X) vec(rho).

For Liouville index mu:  (m_mu, n_mu) = (mu mod 2, mu div 2),  Delta_mu = E_{m_mu} - E_{n_mu}.

Pictures.  rho = e^{-iHt} rho_I e^{iHt}  <=>  vec rho = P(t) vec rho_I,  P = diag(e^{-i Delta_mu t}).
A generator K_I in the interaction picture becomes  K_S(t) = P(t) K_I(t) P(t)^{-1}  in the Schroedinger
picture, and the full Schroedinger generator is  L_H + K_S  with  L_H = -i [H, .].
"""
from __future__ import annotations

import numpy as np

TRACE_ROW = np.array([1.0, 0.0, 0.0, 1.0])   # Tr rho = TRACE_ROW . vec(rho)


def vec(rho: np.ndarray) -> np.ndarray:
    """Column-major vec over the last two axes: (...,2,2) -> (...,4)."""
    rho = np.asarray(rho)
    return np.swapaxes(rho, -1, -2).reshape(rho.shape[:-2] + (4,))


def unvec(v: np.ndarray) -> np.ndarray:
    """Inverse of vec: (...,4) -> (...,2,2)."""
    v = np.asarray(v)
    return np.swapaxes(v.reshape(v.shape[:-1] + (2, 2)), -1, -2)


def sup(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Superoperator of rho -> X rho Y, batched over a leading axis: (n,2,2),(n,2,2) -> (n,4,4)."""
    return np.einsum("tik,tlj->tjilk", X, Y).reshape(X.shape[0], 4, 4)


def kron_RL(R: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Batched  R^T kron L  : the superoperator of rho -> L rho R.  (N,2,2),(N,2,2) -> (N,4,4)."""
    RT = np.swapaxes(R, 1, 2)
    return np.einsum("nij,nkl->nikjl", RT, L).reshape(L.shape[0], 4, 4)


def commutator_generator(E: np.ndarray) -> np.ndarray:
    """L_H = -i [diag(E), .] as a 4x4 matrix."""
    H = np.diag(np.asarray(E, float)).astype(complex)
    I = np.eye(2)
    return -1j * (np.kron(I, H) - np.kron(H.T, I))


def delta_mu(E: np.ndarray) -> np.ndarray:
    """Delta_mu = E_{mu mod 2} - E_{mu div 2} for mu = 0..3."""
    idx = np.arange(4)
    return E[idx % 2] - E[idx // 2]


def to_schrodinger(K: np.ndarray, E: np.ndarray, t) -> np.ndarray:
    """K_S(t) = P(t) K_I(t) P(t)^{-1}; K: (n,4,4) at times t (n,)."""
    t = np.atleast_1d(np.asarray(t, float))
    ph = np.exp(-1j * np.outer(t, delta_mu(E)))            # (n,4)
    return ph[:, :, None] * K * np.conj(ph)[:, None, :]


def to_interaction(K: np.ndarray, E: np.ndarray, t) -> np.ndarray:
    """Inverse of to_schrodinger."""
    t = np.atleast_1d(np.asarray(t, float))
    ph = np.exp(-1j * np.outer(t, delta_mu(E)))
    return np.conj(ph)[:, :, None] * K * ph[:, None, :]


def energy_to_site(K: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Superoperator basis change energy -> site: W K W^{-1},  W = U^* kron U."""
    W = np.kron(U.conj(), U)
    return np.einsum("ij,...jk,kl->...il", W, K, np.linalg.inv(W))
