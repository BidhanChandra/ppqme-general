"""First-order inhomogeneous term for Jang's partial-polaron QME.

Implements Eq. (74) of:
S. J. Jang, J. Chem. Phys. 157, 104107 (2022)

for the N-state independent-bath model used by
``partial_polaron_general.py``.

Conventions
-----------
* hbar = k_B = 1 internally.
* Times are therefore in inverse-energy units (eV^-1 in the SF scripts).
* Site order and eigenvector convention are the same as the existing code:
      U[j,p] = <j|phi_p>
* The existing time-local homogeneous Liouvillian is in the Schrödinger
  picture. Eq. (74) is first constructed in the interaction picture and then
  transformed back to the Schrödinger picture before propagation.

For the usual singlet-fission initial condition |S1><S1| the initial density
matrix is diagonal in the site basis, so the transformed reduced initial state
is identical to the physical initial state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ppqme.core import ComplexArray, PTModel


@dataclass(frozen=True)
class FirstOrderInhomogeneous:
    """First-order source I^(1)(t) sampled on a time grid."""

    times: NDArray[np.float64]
    source_energy: ComplexArray  # (nt,n,n), Schrödinger picture
    f_tensor: ComplexArray       # (n,n,n,nt): f[j,k,kp,t]
    h_tensor: ComplexArray       # (n,n,nt): h[j,kp,t]


def _validate_model_metadata(model: PTModel) -> None:
    required = (
        "h_system",
        "rate_times",
        "phi_site",
        "mixed_site",
        "debye_waller",
    )
    missing = [key for key in required if key not in model.metadata]
    if missing:
        raise KeyError(
            "model metadata is missing fields needed for I^(1): "
            + ", ".join(missing)
        )


def first_order_f_tensor_independent(model: PTModel) -> ComplexArray:
    r"""Compute f_{jk,k'}(t), Eq. (60), for independent baths.

    Eq. (60):
        f_{jk,k'}(t) =
        exp[(2 i/pi) int dω J^(1)_{k',jk}(ω)/ω² W_h(ω)² sin(ωt)]

    For independent baths,
        J^(1)_{k',jk} = (delta_{k'j} - delta_{k'k}) J_{k'}.

    The imaginary part of the already-computed site phi is
        Im phi_{k'}(t)
        = -(1/pi) int dω J_{k'}(ω)/ω² W_h² sin(ωt),

    hence
        f_{jk,k'}(t)
        = exp[-2 i (delta_{k'j}-delta_{k'k}) Im phi_{k'}(t)].
    """
    _validate_model_metadata(model)
    phi = np.asarray(model.metadata["phi_site"], dtype=complex)
    n, nt = phi.shape

    out = np.ones((n, n, n, nt), dtype=complex)
    for j in range(n):
        for k in range(n):
            for kp in range(n):
                coeff = int(kp == j) - int(kp == k)
                if coeff != 0:
                    out[j, k, kp] = np.exp(
                        -2.0j * coeff * np.imag(phi[kp])
                    )
    return out


def first_order_h_tensor_independent(model: PTModel) -> ComplexArray:
    r"""Compute h_{j,k'}(t), Eq. (61), for independent baths.

    Eq. (61):
        h_{j,k'}(t)
        = (2/pi) int dω J_{jk'}(ω)/ω
          [1-W_h(ω)] W_h(ω) cos(ωt).

    For independent baths J_{jk'} = delta_{jk'} J_j and
    Re M_j(t) = (1/pi) int J_j/ω W_h(1-W_h) cos(ωt), therefore

        h_{j,k'}(t) = 2 delta_{jk'} Re M_j(t).
    """
    _validate_model_metadata(model)
    mixed = np.asarray(model.metadata["mixed_site"], dtype=complex)
    n, nt = mixed.shape

    out = np.zeros((n, n, nt), dtype=complex)
    for j in range(n):
        out[j, j] = 2.0 * np.real(mixed[j])
    return out


def _K_independent(
    phi_site: ComplexArray,
    j: int,
    k: int,
    jp: int,
    kp: int,
) -> ComplexArray:
    """K_{jk,j'k'}(t) assembled from independent site baths."""
    out = np.zeros(phi_site.shape[1], dtype=complex)
    if j == jp:
        out += phi_site[j]
    if k == kp:
        out += phi_site[k]
    if j == kp:
        out -= phi_site[j]
    if k == jp:
        out -= phi_site[k]
    return out


def _M_independent(
    mixed_site: ComplexArray,
    j: int,
    jp: int,
    kp: int,
) -> ComplexArray:
    """M_{j,j'k'}(t) assembled from independent site baths."""
    out = np.zeros(mixed_site.shape[1], dtype=complex)
    if j == jp:
        out += mixed_site[j]
    if j == kp:
        out -= mixed_site[j]
    return out


def first_order_bath_trace_tensor_independent(
    model: PTModel,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    r"""Build the bath factor in Eq. (59).

    Returns
    -------
    G
        G[j,k,jp,kp,t] =
        Tr_b{ B~_{jk}(t) delta rho~_{b,jp,kp} }.

    f_tensor, h_tensor
        The Eq. (60) and Eq. (61) functions used to construct G.

    Eq. (59):
      G = w_{j'k'} {
            J~_{jk} [exp(-K_{jk,j'k'}(t)) f_{jk,k'}(t) - 1]
            + delta_jk [M_{j,j'k'}(t) + h_{j,k'}(t)]
          }.

    Here J~_{jk}=J_{jk} w_{jk} for j!=k and zero on the diagonal.
    """
    _validate_model_metadata(model)

    H = np.asarray(model.metadata["h_system"], dtype=complex)
    phi = np.asarray(model.metadata["phi_site"], dtype=complex)
    mixed = np.asarray(model.metadata["mixed_site"], dtype=complex)
    w = np.asarray(model.metadata["debye_waller"], dtype=float)

    n, nt = phi.shape
    if H.shape != (n, n) or w.shape != (n, n):
        raise ValueError("inconsistent model dimensions")

    f_tensor = first_order_f_tensor_independent(model)
    h_tensor = first_order_h_tensor_independent(model)

    Jtilde = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            if j != k:
                Jtilde[j, k] = H[j, k] * w[j, k]

    G = np.zeros((n, n, n, n, nt), dtype=complex)

    for j in range(n):
        for k in range(n):
            for jp in range(n):
                for kp in range(n):
                    K = _K_independent(phi, j, k, jp, kp)
                    value = Jtilde[j, k] * (
                        np.exp(-K) * f_tensor[j, k, kp] - 1.0
                    )

                    if j == k:
                        M = _M_independent(mixed, j, jp, kp)
                        value += M + h_tensor[j, kp]

                    G[j, k, jp, kp] = w[jp, kp] * value

    return G, f_tensor, h_tensor


def first_order_inhomogeneous_source(
    model: PTModel,
    sigma0_site: ArrayLike,
) -> FirstOrderInhomogeneous:
    r"""Evaluate Jang Eq. (74) and return I^(1)(t) in Schrödinger picture.

    Parameters
    ----------
    model
        Partial-polaron PTModel.
    sigma0_site
        Initial *untransformed* reduced system density matrix in the site
        basis, sigma_{j'k'}(0) of Eqs. (54)-(55).

    Notes
    -----
    Eq. (74) is written in the interaction picture. We first evaluate it
    literally and then transform back:
        I_S,pq(t) = exp[-i(E_p-E_q)t] I_I,pq(t).

    No additional factor of hbar appears because the code uses hbar=1.
    """
    _validate_model_metadata(model)

    sigma0 = np.asarray(sigma0_site, dtype=complex)
    E = np.asarray(model.energies, dtype=float)
    U = np.asarray(model.eigenvectors, dtype=complex)
    times = np.asarray(model.metadata["rate_times"], dtype=float)

    n = E.size
    if sigma0.shape != (n, n):
        raise ValueError(f"sigma0_site must have shape {(n,n)}")

    G, f_tensor, h_tensor = first_order_bath_trace_tensor_independent(model)

    nt = times.size
    source_I = np.zeros((nt, n, n), dtype=complex)

    # Literal implementation of Eq. (74):
    #
    # I^(1)_I,pq(t) =
    # -i sum_r sum_jk sum_j'k'
    # sigma_j'k'(0) *
    # [ U*_{jp} U_{kr} U*_{j'r} U_{k'q} exp(i DeltaE_pr t)
    # - U*_{jr} U_{kq} U*_{j'p} U_{k'r} exp(i DeltaE_rq t) ]
    # * G_{jk,j'k'}(t).
    #
    # The w_j'k' factor from Eq. (74) is already contained in G via Eq. (59).

    nonzero_initial = np.argwhere(np.abs(sigma0) > 1.0e-15)

    for it, t in enumerate(times):
        phase = np.exp(1.0j * (E[:, None] - E[None, :]) * t)

        for p in range(n):
            for q in range(n):
                accum = 0.0j

                for r in range(n):
                    for j in range(n):
                        for k in range(n):
                            for jp, kp in nonzero_initial:
                                sig = sigma0[jp, kp]

                                term1 = (
                                    np.conj(U[j, p])
                                    * U[k, r]
                                    * np.conj(U[jp, r])
                                    * U[kp, q]
                                    * phase[p, r]
                                )

                                term2 = (
                                    np.conj(U[j, r])
                                    * U[k, q]
                                    * np.conj(U[jp, p])
                                    * U[kp, r]
                                    * phase[r, q]
                                )

                                accum += (
                                    sig
                                    * (term1 - term2)
                                    * G[j, k, jp, kp, it]
                                )

                source_I[it, p, q] = -1.0j * accum

    # Eq. (74) is in the interaction picture; convert to Schrödinger picture.
    source_S = np.empty_like(source_I)
    for it, t in enumerate(times):
        phase_back = np.exp(
            -1.0j * (E[:, None] - E[None, :]) * t
        )
        source_S[it] = phase_back * source_I[it]

    return FirstOrderInhomogeneous(
        times=times,
        source_energy=source_S,
        f_tensor=f_tensor,
        h_tensor=h_tensor,
    )


def downsample_first_order_source(
    first_order: FirstOrderInhomogeneous,
    indices: ArrayLike,
) -> FirstOrderInhomogeneous:
    """Downsample a first-order source using indices from the fine grid."""
    idx = np.asarray(indices, dtype=int)
    return FirstOrderInhomogeneous(
        times=first_order.times[idx],
        source_energy=first_order.source_energy[idx],
        f_tensor=first_order.f_tensor[..., idx],
        h_tensor=first_order.h_tensor[..., idx],
    )


def propagate_time_local_affine_piecewise_midpoint(
    grid_times: ArrayLike,
    liouvillians: ArrayLike,
    source_matrices: ArrayLike,
    rho0: ArrayLike,
) -> ComplexArray:
    r"""Propagate d rho/dt = L(t) rho + I(t).

    A midpoint exponential is applied to an augmented state vector:
        d/dt [vec(rho)] = L vec(rho) + vec(I)
        d/dt [1       ] = 0.

    This handles the affine source without assuming L is invertible.
    """
    t = np.asarray(grid_times, dtype=float)
    L = np.asarray(liouvillians, dtype=complex)
    src = np.asarray(source_matrices, dtype=complex)
    rho0 = np.asarray(rho0, dtype=complex)

    n = rho0.shape[0]
    n2 = n * n

    if rho0.shape != (n, n):
        raise ValueError("rho0 must be square")
    if L.shape != (t.size, n2, n2):
        raise ValueError("liouvillians have incompatible shape")
    if src.shape != (t.size, n, n):
        raise ValueError("source_matrices have incompatible shape")

    x = np.empty(n2 + 1, dtype=complex)
    x[:n2] = rho0.reshape(-1, order="F")
    x[n2] = 1.0

    states = np.empty((t.size, n, n), dtype=complex)
    states[0] = rho0

    for it in range(t.size - 1):
        dt = t[it + 1] - t[it]

        Lmid = 0.5 * (L[it] + L[it + 1])
        Imid = 0.5 * (src[it] + src[it + 1])

        A = np.zeros((n2 + 1, n2 + 1), dtype=complex)
        A[:n2, :n2] = Lmid
        A[:n2, n2] = Imid.reshape(-1, order="F")

        x = expm(A * dt) @ x
        states[it + 1] = x[:n2].reshape((n, n), order="F")

    return states


def first_order_source_diagnostics(
    first_order: FirstOrderInhomogeneous,
) -> dict[str, float]:
    """Trace/Hermiticity diagnostics for I^(1)(t)."""
    src = np.asarray(first_order.source_energy, dtype=complex)

    trace_err = float(
        np.max(
            np.abs(
                np.trace(src, axis1=1, axis2=2)
            )
        )
    )

    herm_err = float(
        np.max(
            np.abs(
                src - np.swapaxes(src.conj(), 1, 2)
            )
        )
    )

    norm = np.linalg.norm(
        src.reshape(src.shape[0], -1),
        axis=1,
    )

    return {
        "max_abs_trace_I1": trace_err,
        "max_hermiticity_error_I1": herm_err,
        "max_frobenius_norm_I1": float(np.max(norm)),
        "time_index_max_norm_I1": int(np.argmax(norm)),
    }
