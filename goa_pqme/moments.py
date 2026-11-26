r"""Exact multi-time moments  < B_{a_1}(t_1) B_{a_2}(t_2) ... B_{a_n}(t_n) >  of the residual bath operators.

The bath is harmonic and Gaussian, but B_DA, B_AD = Gamma (D_s - B) are exponentials, so their odd moments do not
vanish (this is why K3 != 0).  Each hopping factor is expanded  Gamma (D_s - B) -> Gamma D_s  or  -Gamma B,
which leaves ordered products of D_s exponentials and linear operators X.  For such a product (items i < j in
operator order, tau_ij = t_j - t_i) the exact Gaussian result is

    < ... > = exp(c0) * sum over partial pairings of the X's of  prod_pairs G_ij  prod_unpaired l_i

    c0  = - n_D * phi(0)/2  -  sum_{i<j, both D} s_i s_j g_DD(tau_ij)
    l_i = sum_{D_j after X_i} s_j h_DX(tau_ij)  -  sum_{D_j before X_i} s_j h_DX(tau_ji)
    G_ij = g_XX(tau_ij)
with the pair functions (kappa in DD, DX, XX; weights w^DD = xi^2, w^DX = xi x, w^XX = x^2)
    g_kappa(tau) = sum_q w_q [ c_q cos(w_q tau) + i sin(w_q tau) ]
    h_kappa(tau) = sum_q w_q [ cos(w_q tau) + i c_q sin(w_q tau) ]

Implementation note: g and h are formed from the two tables
    C_kappa(tau) = sum_q w_q e^{i w_q tau},   T_kappa(tau) = sum_q (n_q + 1/2) w_q e^{i w_q tau},
as  alpha = T(-tau) - C(tau)^*/2,  beta = C(tau)/2 + T(tau),  g = alpha + beta,  h = beta - alpha.
Two-point special cases reproduce the TCL2 table, e.g.  <D_-(t) D_+(0)> = B^2 e^{+phi(t)}.
"""
from __future__ import annotations

import itertools

import numpy as np

from .frame import PolaronFrame

_KIND = {"DD": 0, "DX": 1, "XD": 1, "XX": 2}


class ResidualBath:
    """Gaussian moment engine for one frame (the bath part of the old fastk.FastTCL)."""

    def __init__(self, frame: PolaronFrame):
        xi, x, nh = frame.xi, frame.x, frame.nplus_half
        self.w = frame.omega
        self.Gamma = complex(frame.Gamma)
        self.B = frame.B
        wt = (xi * xi, xi * x, x * x)
        self.wC = np.stack(wt, 1)                     # (M,3)  weights for C_kappa
        self.wT = np.stack([nh * k for k in wt], 1)   # (M,3)  weights for T_kappa
        self.T0DD = float(np.sum(nh * xi * xi))       # = phi(0)/2

    # -- pair tables for all ordered pairs of columns of T ---------------------------------
    def tables(self, T: np.ndarray):
        """T: (N, n) times.  Returns dicts C[(p,q)], Tt[(p,q)] of (N,3) arrays at tau = T[:,q] - T[:,p]."""
        N, n = T.shape
        C, Tt = {}, {}
        for p in range(n):
            for q in range(p + 1, n):
                E = np.exp(1j * np.outer(T[:, q] - T[:, p], self.w))
                c, t = E @ self.wC, E @ self.wT
                C[(p, q)], C[(q, p)] = c, np.conj(c)
                Tt[(p, q)], Tt[(q, p)] = t, np.conj(t)
        return C, Tt

    # -- one ordered sequence of D / X items -------------------------------------------------
    def sequence_moment(self, items, C, Tt, N: int) -> np.ndarray:
        """items: tuple of (kind, sign, column) with kind 'D' (sign +-1) or 'X'."""
        n = len(items)
        xs = [i for i, it in enumerate(items) if it[0] == "X"]
        pos = {i: k for k, i in enumerate(xs)}
        lin = [np.zeros(N, complex) for _ in xs]
        G = {}
        c0 = np.zeros(N, complex) - self.T0DD * sum(1 for it in items if it[0] == "D")
        for i in range(n):
            ki, si, p = items[i]
            for j in range(i + 1, n):
                kj, sj, q = items[j]
                k = _KIND[ki + kj]
                Cpq = C[(p, q)][:, k]
                alpha = Tt[(q, p)][:, k] - 0.5 * np.conj(Cpq)
                beta = 0.5 * Cpq + Tt[(p, q)][:, k]
                if ki == "D" and kj == "D":
                    c0 = c0 - si * sj * (alpha + beta)
                elif ki == "X" and kj == "D":
                    lin[pos[i]] = lin[pos[i]] + sj * (beta - alpha)
                elif ki == "D" and kj == "X":
                    lin[pos[j]] = lin[pos[j]] + si * (alpha - beta)
                else:
                    G[(pos[i], pos[j])] = alpha + beta
        F0 = np.exp(c0)
        return F0 if not xs else F0 * self._wick(tuple(range(len(xs))), lin, G)

    def _wick(self, idx, lin, G):
        """Sum over partial pairings: W(I) = l_i W(I\\i) + sum_j G_ij W(I\\{i,j})."""
        if not idx:
            return 1.0
        i, rest = idx[0], idx[1:]
        val = lin[i] * self._wick(rest, lin, G)
        for k, j in enumerate(rest):
            g = G[(i, j)] if (i, j) in G else G[(j, i)]
            val = val + g * self._wick(rest[:k] + rest[k + 1:], lin, G)
        return val

    # -- moment of residual operators ---------------------------------------------------------
    def moment(self, channels, columns, C, Tt, N: int) -> np.ndarray:
        """< B_{channels[0]}(T[:,columns[0]]) B_{channels[1]}(...) ... >, channels in {1:AA, 2:DA, 3:AD}."""
        hop = [k for k, ch in enumerate(channels) if ch in (2, 3)]
        total = np.zeros(N, complex)
        for choose in itertools.product((False, True), repeat=len(hop)):
            coeff = self.Gamma ** len(hop)
            items, h = [], 0
            for ch, p in zip(channels, columns):
                if ch == 1:
                    items.append(("X", 0.0, p))
                else:
                    if choose[h]:
                        items.append(("D", -1.0 if ch == 2 else 1.0, p))
                    else:
                        coeff = coeff * (-self.B)
                    h += 1
            total = total + coeff * self.sequence_moment(tuple(items), C, Tt, N)
        return total

    def moment_at(self, channels, times) -> complex:
        """Convenience: one moment at explicit times, e.g. moment_at((2,3), (t, 0.0))."""
        T = np.asarray(times, float)[None, :]
        C, Tt = self.tables(T)
        return complex(self.moment(tuple(channels), tuple(range(len(channels))), C, Tt, 1)[0])
