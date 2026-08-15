"""General N-state independent-bath p-PQME model builder."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
from .core import PTModel, ResidualChannel
from .bath import BathNumerics, SpectralDensity, independent_bath_components_grid
from .transform import PartialPolaronTransform


def make_channel_labels(H, threshold=1e-14):
    H = np.asarray(H, complex); n = H.shape[0]
    labels = [(j,j) for j in range(n)]
    labels += [(j,k) for j in range(n) for k in range(n) if j != k and abs(H[j,k]) > threshold]
    return labels


def _K(phi, j,k,jp,kp):
    out = np.zeros(phi.shape[1], complex)
    if j==jp: out += phi[j]
    if k==kp: out += phi[k]
    if j==kp: out -= phi[j]
    if k==jp: out -= phi[k]
    return out


def _M(mixed, j,jp,kp):
    out = np.zeros(mixed.shape[1], complex)
    if j==jp: out += mixed[j]
    if j==kp: out -= mixed[j]
    return out


def debye_waller_matrix(phi0):
    phi0 = np.asarray(phi0, float); n = phi0.size
    w = np.ones((n,n))
    for j in range(n):
        for k in range(n):
            if j != k:
                w[j,k] = np.exp(-0.5*(phi0[j]+phi0[k]))
    return w


def transformed_hamiltonian(H, lambda_partial, w):
    H = np.asarray(H, complex); n = H.shape[0]
    out = H.copy()
    for j in range(n): out[j,j] = H[j,j]-lambda_partial[j]
    for j in range(n):
        for k in range(n):
            if j != k: out[j,k] = H[j,k]*w[j,k]
    return 0.5*(out+out.conj().T)


def correlation_tensor_independent(H, labels, phi, mixed, linear, w):
    H = np.asarray(H, complex); n,nt = phi.shape
    Jt = np.zeros((n,n), complex)
    for j in range(n):
        for k in range(n):
            if j != k: Jt[j,k] = H[j,k]*w[j,k]
    nc = len(labels); corr = np.zeros((nc,nc,nt), complex)
    for a,(j,k) in enumerate(labels):
        ad = (j==k)
        for b,(jp,kp) in enumerate(labels):
            bd = (jp==kp); val = np.zeros(nt, complex)
            if not ad and not bd:
                val += Jt[j,k]*Jt[jp,kp]*np.expm1(-_K(phi,j,k,jp,kp))
            if ad and not bd:
                val += Jt[jp,kp]*_M(mixed,j,jp,kp)
            if not ad and bd:
                val += Jt[j,k]*_M(mixed,jp,k,j)
            if ad and bd and j==jp:
                val += linear[j]
            corr[a,b] = val
    return corr


def _fix_phases(U):
    U = U.copy()
    for p in range(U.shape[1]):
        q = int(np.argmax(np.abs(U[:,p])))
        U[:,p] *= np.exp(-1j*np.angle(U[q,p]))
        if U[q,p].real < 0: U[:,p] *= -1
    return U


def build_model(H: ArrayLike, spectral_densities: Sequence[SpectralDensity], numerics: BathNumerics, transform: PartialPolaronTransform, channel_threshold=1e-14) -> PTModel:
    numerics.validate(); transform.validate()
    H = np.asarray(H, complex); n = H.shape[0]
    if H.shape != (n,n) or not np.allclose(H,H.conj().T):
        raise ValueError('H must be Hermitian and square')
    nt = int(round(numerics.rate_tmax/numerics.rate_dt))+1
    times = np.linspace(0,numerics.rate_tmax,nt)
    bath = independent_bath_components_grid(times, spectral_densities, numerics.beta, transform)
    phi = bath['phi_site']; mixed=bath['mixed_site']; linear=bath['linear_site']
    lam = bath['lambda_partial']; w = debye_waller_matrix(np.real(phi[:,0]))
    Ht = transformed_hamiltonian(H, lam, w)
    E,U = np.linalg.eigh(Ht); U = _fix_phases(U)
    labels = make_channel_labels(H, channel_threshold)
    channels=[]
    for j,k in labels:
        op = np.zeros((n,n), complex); op[j,k]=1
        channels.append(ResidualChannel((j,k), U.conj().T@op@U, float(w[j,k]) if j!=k else 1.0))
    corr = correlation_tensor_independent(H, labels, phi, mixed, linear, w)
    corr_neg = correlation_tensor_independent(H, labels, phi.conj(), mixed.conj(), linear.conj(), w)
    return PTModel(Ht,E,U,channels,numerics.beta,dict(
        h_system=H, rate_times=times, phi_site=phi, mixed_site=mixed,
        linear_site=linear, lambda_partial=lam, debye_waller=w,
        correlation_tensor=corr, correlation_tensor_negative=corr_neg,
        channel_labels=labels, omega_nodes=bath['omega_nodes'],
        transformed_fraction=bath['transformed_fraction'], transform=transform,
        numerics=numerics))
