"""Second-order inhomogeneous term for Jang's partial-polaron QME.

Implements Eq. (76) and Appendix-C kernels (C8, C17, C18, C27) of

    S. J. Jang, J. Chem. Phys. 157, 104107 (2022)

for the independent-bath N-state model used in the accompanying SF code.

IMPORTANT
---------
This implementation is specialized to a *site-localized initial population*
|j0><j0|.  That is exactly the singlet-fission initial condition used here:
|S1><S1| with j0 = 0.

That specialization is valuable because j'' = k'' = j0 in Eq. (76), and the
Appendix-C kernels simplify substantially:
    K_{jk,j0j0} = 0
    M_{j,j0j0} = 0
    w_{j0j0} = 1.

The homogeneous finite-time rates should still be constructed on a fine grid
(e.g. 0.1 fs).  I^(2) is more expensive; calculate it on the propagation grid
(e.g. 0.5 fs) and converge separately with 0.25 fs if needed.

Conventions
-----------
* hbar = 1 internally.
* Times are inverse-energy units (eV^-1 in the SF scripts).
* U[j,p] = <j|phi_p>.
* Eq. (76) is evaluated in the interaction picture and then transformed back
  to the Schrödinger picture used by the existing propagator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ppqme.core import ComplexArray, PTModel
from advanced.inhomogeneous_first_order import (
    first_order_f_tensor_independent,
    first_order_h_tensor_independent,
)


@dataclass(frozen=True)
class SecondOrderInhomogeneous:
    times: NDArray[np.float64]
    source_energy: ComplexArray  # (nt,n,n), Schrödinger picture
    active_channels: tuple[tuple[int, int], ...]


def _K_independent(
    phi_site: ComplexArray,
    j: int,
    k: int,
    jp: int,
    kp: int,
) -> ComplexArray:
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
    out = np.zeros(mixed_site.shape[1], dtype=complex)
    if j == jp:
        out += mixed_site[j]
    if j == kp:
        out -= mixed_site[j]
    return out


def _trapz(y: ComplexArray, x: NDArray[np.float64], axis: int = -1):
    """NumPy-version-compatible trapezoidal integral."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def active_residual_channels(model: PTModel) -> tuple[tuple[int, int], ...]:
    """Site-operator channels that can contribute to Eq. (76).

    A diagonal (j,j) channel contributes through D_j.
    An off-diagonal (j,k) channel contributes only when bare J_jk != 0.
    """
    H = np.asarray(model.metadata["h_system"], dtype=complex)
    n = H.shape[0]
    channels: list[tuple[int, int]] = []

    for j in range(n):
        channels.append((j, j))

    for j in range(n):
        for k in range(n):
            if j != k and abs(H[j, k]) > 1.0e-15:
                channels.append((j, k))

    return tuple(channels)


def _sample_bath_objects(
    model: PTModel,
    sample_indices: NDArray[np.int64],
):
    """Return bath arrays sampled on the I2/propagation grid."""
    phi = np.asarray(model.metadata["phi_site"], dtype=complex)[..., sample_indices]
    mixed = np.asarray(model.metadata["mixed_site"], dtype=complex)[..., sample_indices]
    w = np.asarray(model.metadata["debye_waller"], dtype=float)

    f_full = first_order_f_tensor_independent(model)
    h_full = first_order_h_tensor_independent(model)

    f = f_full[..., sample_indices]
    h = h_full[..., sample_indices]

    return phi, mixed, w, f, h


def _appendix_c_kernel_values_local_initial(
    *,
    phi: ComplexArray,
    mixed: ComplexArray,
    w: NDArray[np.float64],
    f: ComplexArray,
    h: ComplexArray,
    j0: int,
    a: tuple[int, int],
    b: tuple[int, int],
    it: int,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    r"""Return F, H1, H2, L as functions of tau for fixed current time t.

    This is Eqs. (C8), (C17), (C18), and (C27), specialized to j''=k''=j0.

    Returned arrays have length it+1 and correspond to tau = times[:it+1].
    """
    j, k = a
    jp, kp = b

    tau_slice = slice(0, it + 1)
    m = it + 1

    # s = t - tau.  On a uniform grid this is represented by reversed indices.
    # All grids used by the runner are uniform.
    s_idx = np.arange(it, -1, -1, dtype=int)

    # K_{jk,j'k'}(t-tau)
    K_ab_s_full = _K_independent(phi, j, k, jp, kp)
    K_ab_s = K_ab_s_full[s_idx]

    # Because the initial pair is j0,j0:
    # K_{jk,j0j0}=0 and M_{j,j0j0}=0 identically.
    f_a0_t = f[j, k, j0, it]
    f_b0_tau = f[jp, kp, j0, tau_slice]

    wa = w[j, k]
    wb = w[jp, kp]

    # Eq. (C8), w_{j0,j0}=1 and both cross K's to j0j0 vanish.
    F = wa * wb * (
        np.exp(-K_ab_s) * (f_a0_t * f_b0_tau - 1.0)
        - f_a0_t
        - f_b0_tau
        + 2.0
    )

    # Eq. (C17):
    # (f_b0(tau)-1) wb [M_{j,b}(t-tau)+h_{j,j0}(t)]
    M_j_b_full = _M_independent(mixed, j, jp, kp)
    M_j_b_s = M_j_b_full[s_idx]
    H1 = (
        (f_b0_tau - 1.0)
        * wb
        * (M_j_b_s + h[j, j0, it])
    )

    # Eq. (C18):
    # (f_a0(t)-1) wa [M_{j',kj}(t-tau)+h_{j',j0}(tau)]
    M_jp_kj_full = _M_independent(mixed, jp, k, j)
    M_jp_kj_s = M_jp_kj_full[s_idx]
    H2 = (
        (f_a0_t - 1.0)
        * wa
        * (M_jp_kj_s + h[jp, j0, tau_slice])
    )

    # Eq. (C27): cross M's to the diagonal initial pair vanish.
    L = h[j, j0, it] * h[jp, j0, tau_slice]

    return F, H1, H2, L


def second_order_inhomogeneous_source_local_initial(
    model: PTModel,
    sample_indices: ArrayLike,
    *,
    initial_site: int = 0,
    progress: bool = True,
) -> SecondOrderInhomogeneous:
    r"""Evaluate Jang Eq. (76) for sigma(0)=|initial_site><initial_site|.

    Parameters
    ----------
    model
        Existing partially-polaron-transformed model.
    sample_indices
        Indices into model.metadata["rate_times"].  I2 is evaluated only on
        this grid.  Use the same indices as the propagation grid.
    initial_site
        Site j0 of the localized initial population.
    progress
        Print coarse progress messages.

    Returns
    -------
    SecondOrderInhomogeneous
        Schrödinger-picture I^(2)(t).

    Notes
    -----
    For each current time t, the Appendix-C kernels are integrated over tau
    using Eqs. (C28)-(C31).  The four kernels are combined with the bare
    electronic couplings exactly as in Eq. (76).
    """
    idx = np.asarray(sample_indices, dtype=int)
    fine_times = np.asarray(model.metadata["rate_times"], dtype=float)
    times = fine_times[idx]

    if times.size < 2:
        raise ValueError("Need at least two time points.")

    dt = np.diff(times)
    if not np.allclose(dt, dt[0], rtol=1e-8, atol=1e-12):
        raise ValueError(
            "This implementation currently requires a uniform sampled grid."
        )

    E = np.asarray(model.energies, dtype=float)
    U = np.asarray(model.eigenvectors, dtype=complex)
    Hsys = np.asarray(model.metadata["h_system"], dtype=complex)

    n = E.size
    if not (0 <= initial_site < n):
        raise ValueError("initial_site out of range")

    phi, mixed, w, f, h = _sample_bath_objects(model, idx)
    channels = active_residual_channels(model)
    nch = len(channels)
    nt = times.size

    # Bare electronic hopping J_jk in Eq. (76); by definition diagonal is zero.
    J = np.array(Hsys, copy=True)
    np.fill_diagonal(J, 0.0)

    # Integrated bath object for each channel pair and each energy gap E_a-E_b:
    #
    # Bint[it, ia, ib, pprime, r]
    #
    # = J_a J_b F~ + delta_a J_b H1~ +
    #   J_a delta_b H2~ + delta_a delta_b L~
    #
    # evaluated at E = E_pprime - E_r.
    Bint = np.zeros((nt, nch, nch, n, n), dtype=complex)

    gap = E[:, None] - E[None, :]

    for it in range(1, nt):
        if progress and (
            it == 1
            or it == nt - 1
            or it % max(1, nt // 10) == 0
        ):
            print(
                f"I2 bath integrals: {it:5d}/{nt-1:5d} "
                f"(t = {times[it]:.3f} eV^-1)"
            )

        tau = times[:it + 1]
        s = times[it] - tau

        # phase[p',r,tau]
        phase = np.exp(
            1.0j * gap[:, :, None] * s[None, None, :]
        )

        for ia, a in enumerate(channels):
            j, k = a
            delta_a = int(j == k)
            Ja = J[j, k]

            for ib, b in enumerate(channels):
                jp, kp = b
                delta_b = int(jp == kp)
                Jb = J[jp, kp]

                F, H1, H2, L = _appendix_c_kernel_values_local_initial(
                    phi=phi,
                    mixed=mixed,
                    w=w,
                    f=f,
                    h=h,
                    j0=initial_site,
                    a=a,
                    b=b,
                    it=it,
                )

                kernel = (
                    Ja * Jb * F
                    + delta_a * Jb * H1
                    + Ja * delta_b * H2
                    + delta_a * delta_b * L
                )

                if np.max(np.abs(kernel)) < 1.0e-18:
                    continue

                # Integrate all n^2 energy gaps at once.
                Bint[it, ia, ib] = _trapz(
                    phase * kernel[None, None, :],
                    x=tau,
                    axis=-1,
                )

    # Eq. (76), specialized to sigma_{j''k''}(0)=delta_{j'',j0}delta_{k'',j0}.
    source_I_core = np.zeros((nt, n, n), dtype=complex)

    for it, t in enumerate(times):
        if progress and (
            it == 0
            or it == nt - 1
            or it % max(1, nt // 10) == 0
        ):
            print(
                f"I2 Eq.(76):       {it:5d}/{nt-1:5d}"
            )

        phase = np.exp(
            1.0j * (E[:, None] - E[None, :]) * t
        )

        for p in range(n):
            for q in range(n):
                accum = 0.0j

                for pp in range(n):
                    for qp in range(n):
                        initial_U = (
                            np.conj(U[initial_site, pp])
                            * U[initial_site, qp]
                        )

                        if abs(initial_U) < 1.0e-15:
                            continue

                        phase_pp = phase[p, pp]
                        phase_qq = np.exp(
                            -1.0j * (E[q] - E[qp]) * t
                        )

                        for r in range(n):
                            # The bath integral uses E_pp - E_r.
                            for ia, (j, k) in enumerate(channels):
                                for ib, (jp, kp) in enumerate(channels):
                                    B = Bint[it, ia, ib, pp, r]
                                    if abs(B) < 1.0e-18:
                                        continue

                                    term1 = 0.0j
                                    if qp == q:
                                        term1 = (
                                            np.conj(U[j, p])
                                            * U[k, r]
                                            * U[jp, r]
                                            * U[kp, pp]
                                        )

                                    term2 = 0.0j
                                    if r == p:
                                        term2 = (
                                            np.conj(U[j, qp])
                                            * U[k, q]
                                            * np.conj(U[jp, p])
                                            * U[kp, pp]
                                            * phase_qq
                                        )

                                    coeff = (
                                        initial_U
                                        * phase_pp
                                        * (term1 - term2)
                                    )

                                    accum += coeff * B

                # Overall minus sign of Eq. (76), before H.c.
                source_I_core[it, p, q] = -accum

    # Eq. (76): "+ H.c." of the preceding contribution.
    source_I = (
        source_I_core
        + np.swapaxes(source_I_core.conj(), 1, 2)
    )

    # Convert interaction-picture source to Schrödinger picture.
    source_S = np.empty_like(source_I)
    for it, t in enumerate(times):
        phase_back = np.exp(
            -1.0j * (E[:, None] - E[None, :]) * t
        )
        source_S[it] = phase_back * source_I[it]

    return SecondOrderInhomogeneous(
        times=times,
        source_energy=source_S,
        active_channels=channels,
    )


def second_order_source_diagnostics(
    second_order: SecondOrderInhomogeneous,
) -> dict[str, float]:
    src = np.asarray(second_order.source_energy, dtype=complex)

    traces = np.trace(src, axis1=1, axis2=2)
    herm = src - np.swapaxes(src.conj(), 1, 2)
    norm = np.linalg.norm(
        src.reshape(src.shape[0], -1),
        axis=1,
    )

    return {
        "max_abs_trace_I2": float(np.max(np.abs(traces))),
        "max_hermiticity_error_I2": float(np.max(np.abs(herm))),
        "max_frobenius_norm_I2": float(np.max(norm)),
        "time_index_max_norm_I2": int(np.argmax(norm)),
    }
