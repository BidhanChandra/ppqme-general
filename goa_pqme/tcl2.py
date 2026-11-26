r"""TCL2: residual correlation tensor C_ab(t), finite-time rate tensors, Schroedinger generator L(t).

Channel correlation tensor  C_ab(t) = <B_a(t) B_b(0)>,  channel order (DD, AA, DA, AD), a~ = Gamma B:

            AA            DA                       AD
    AA [ linear       -a~ mixed               +a~ mixed            ]
    DA [ +a~ mixed     a~^2 (e^{-phi} - 1)     a~^2 (e^{+phi} - 1)  ]
    AD [ -a~ mixed     a~^2 (e^{+phi} - 1)     a~^2 (e^{-phi} - 1)  ]      (DD row/column = 0)

Backward tensor  C^-_ba(t) = <B_b(-t) B_a(0)> = the same table with phi, mixed, linear -> complex conjugates.

Finite-time rates (Delta_mn = E_m - E_n, cumulative trapezoid on the grid t):
    Wf_{ab,mn}(t) = int_0^t ds C_ab(s)   e^{-i Delta_mn s},     Wb_{ab,mn}(t) = int_0^t ds C^-_ab(s) e^{-i Delta_mn s}

With Lambda_a = sum_b Wf_ab o S_b   (o = elementwise product, = int_0^t ds sum_b C_ab(s) S_b(-s))
and  Lambda'_a = sum_b Wb_ba o S_b:
    d rho/dt = -i[H~_S, rho] + sum_a ( Lambda_a rho S_a - S_a Lambda_a rho - rho Lambda'_a S_a + S_a rho Lambda'_a )
In the interaction picture the dissipative part equals  K_2(t) = int_0^t dt1 <L(t) L(t1)>  (see tcl_higher).
"""
from __future__ import annotations

import numpy as np

from .frame import PolaronFrame
from .liouville import sup


def correlation_tensor_from(phi, mixed, linear, a_tilde: float) -> np.ndarray:
    """Assemble C_ab(t) (4,4,nt) from the three primitive functions (closed form above)."""
    nt = len(phi)
    C = np.zeros((4, 4, nt), complex)
    C[1, 1] = linear
    C[1, 2], C[1, 3] = -a_tilde * mixed, a_tilde * mixed
    C[2, 1], C[3, 1] = a_tilde * mixed, -a_tilde * mixed
    C[2, 2] = C[3, 3] = a_tilde ** 2 * np.expm1(-phi)
    C[2, 3] = C[3, 2] = a_tilde ** 2 * np.expm1(phi)
    return C


def correlation_tensors(frame: PolaronFrame, t) -> tuple[np.ndarray, np.ndarray]:
    """(C, C^-) on the grid t, each (4,4,nt)."""
    phi, mixed, linear = frame.phi(t), frame.mixed(t), frame.linear(t)
    a = frame.Gamma * frame.B
    return (correlation_tensor_from(phi, mixed, linear, a),
            correlation_tensor_from(np.conj(phi), np.conj(mixed), np.conj(linear), a))


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid along the last axis, starting at 0."""
    out = np.zeros_like(y, dtype=complex)
    if len(x) > 1:
        out[..., 1:] = np.cumsum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1)
    return out


def rate_tensors(E: np.ndarray, t, C: np.ndarray, Cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wf, Wb of shape (nt,4,4,2,2), index [t, a, b, m, n]."""
    t = np.asarray(t, float)
    delta = E[:, None] - E[None, :]
    phase = np.exp(-1j * delta[:, :, None] * t[None, None, :])                 # (2,2,nt)
    Wf = cumulative_trapezoid(C[:, :, None, None, :] * phase[None, None], t)
    Wb = cumulative_trapezoid(Cm[:, :, None, None, :] * phase[None, None], t)
    return np.moveaxis(Wf, -1, 0), np.moveaxis(Wb, -1, 0)


def generator_from_rates(E: np.ndarray, S: np.ndarray, Wf: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    """Schroedinger-picture TCL2 generator L(t), (nt,4,4), energy basis, column-major vec."""
    nt = Wf.shape[0]
    I = np.broadcast_to(np.eye(2, dtype=complex), (nt, 2, 2))
    H = np.broadcast_to(np.diag(E).astype(complex), (nt, 2, 2))
    L = -1j * (sup(H, I) - sup(I, H))
    for a in range(4):
        Sa = np.broadcast_to(S[a], (nt, 2, 2))
        Lam = np.einsum("tbmn,bmn->tmn", Wf[:, a], S)          # Lambda_a(t)
        Lamp = np.einsum("tbmn,bmn->tmn", Wb[:, :, a], S)      # Lambda'_a(t)
        SL = np.einsum("ij,tjk->tik", S[a], Lam)
        LS = np.einsum("tij,jk->tik", Lamp, S[a])
        L += sup(Lam, Sa) - sup(SL, I) - sup(I, LS) + sup(Sa, Lamp)
    return L


class TCL2:
    """TCL2 objects of one frame on one time grid (attributes C, Cm, Wf, Wb, L)."""

    def __init__(self, frame: PolaronFrame, t):
        self.frame = frame
        self.t = np.asarray(t, float)
        self.C, self.Cm = correlation_tensors(frame, self.t)
        self.Wf, self.Wb = rate_tensors(frame.E, self.t, self.C, self.Cm)
        self.L = generator_from_rates(frame.E, frame.S, self.Wf, self.Wb)


def tcl2_generator(frame: PolaronFrame, t) -> np.ndarray:
    return TCL2(frame, t).L
