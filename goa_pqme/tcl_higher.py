r"""Time-convolutionless generators K2, K3, K4 (interaction picture, energy basis) at sparse time nodes.

Interaction picture:  L(t) . = -i [H~_I(t), .],   [S_a(t)]_mn = e^{i (E_m - E_n) t} [S_a]_mn.

Liouville moments (P = Tr_B projection onto rho_B; ordering t_0 >= t_1 >= ...):
    M_n(t_0..t_{n-1}) = P L(t_0) ... L(t_{n-1}) P
       = sum_{a in {AA,DA,AD}^n} sum_{branches in {L,R}^n} (-i)^{n_L} (+i)^{n_R}
         < B_{right ops, descending k}  B_{left ops, ascending k} >  (R^T kron L)
    where L = product of the left-branch S_{a_k}(t_k) (ascending k) and R = right-branch product (descending k).
For n = 2 this is explicitly (tau = t - t1)
    M_2 rho = sum_ab { C_ab(tau) [S_b(t1) rho S_a(t) - S_a(t) S_b(t1) rho]
                      + C^-_ba(tau) [S_a(t) rho S_b(t1) - rho S_b(t1) S_a(t)] }.

Ordered cumulants for P L P = 0:
    K2(t) = int_0^t dt1 M2(t,t1)
    K3(t) = int_0^t dt1 int_0^t1 dt2 M3(t,t1,t2)
    K4(t) = int_0^t dt1 int_0^t1 dt2 int_0^t2 dt3 [ M4 - M2(t,t1)M2(t2,t3) - M2(t,t2)M2(t1,t3) - M2(t,t3)M2(t1,t2) ]

Quadrature: nested composite Gauss-Legendre, nq points on ceil(L/h) equal panels of every inner interval.
Generator-size ratio:  r_n = rms_k ||K_n(t_k)||_F / rms_k ||K_2(t_k)||_F  over the non-zero nodes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .config import ACTIVE
from .frame import PolaronFrame
from .liouville import kron_RL
from .moments import ResidualBath

PAIRS_K4 = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))


# ---------------------------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------------------------
def gauss_panels(a: float, b: float, nq: int, h: float | None):
    """Composite Gauss-Legendre rule on [a,b]: m = max(1, ceil((b-a)/h)) equal panels, nq nodes each."""
    x, w = np.polynomial.legendre.leggauss(int(nq))
    L = b - a
    m = 1 if h is None else max(1, int(np.ceil(L / h - 1e-12)))
    xs, ws = [], []
    for k in range(m):
        a0, b0 = a + L * k / m, a + L * (k + 1) / m
        xs.append(0.5 * (b0 - a0) * x + 0.5 * (a0 + b0))
        ws.append(0.5 * (b0 - a0) * w)
    return np.concatenate(xs), np.concatenate(ws)


def nested_nodes(t: float, depth: int, nq: int, h: float | None):
    """Tuples (t, t1, ..., t_depth) with t > t1 > ... > t_depth > 0 and product weights: T (N, depth+1), W (N,)."""
    rows = [((float(t),), 1.0)]
    for _ in range(depth):
        new = []
        for tup, wt in rows:
            xs, ws = gauss_panels(0.0, tup[-1], nq, h)
            new.extend((tup + (x,), wt * wx) for x, wx in zip(xs, ws))
        rows = new
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


# ---------------------------------------------------------------------------------------------
# Liouville moments
# ---------------------------------------------------------------------------------------------
class MomentEngine:
    """Evaluates M_n on batches of time tuples for one frame."""

    def __init__(self, frame: PolaronFrame):
        self.frame = frame
        self.E = frame.E
        self.S = frame.S
        self.bath = ResidualBath(frame)

    def liouville_moment(self, T: np.ndarray, by_assignment: bool = False):
        """M_n for every row of T (N, n).  Returns (N,4,4), or {assignment tuple: (N,4,4)} if by_assignment."""
        N, n = T.shape
        C, Tt = self.bath.tables(T)
        dE = (self.E[:, None] - self.E[None, :])[None]
        St = [{a: self.S[a][None] * np.exp(1j * dE * T[:, k][:, None, None]) for a in ACTIVE} for k in range(n)]
        eye = np.broadcast_to(np.eye(2, dtype=complex), (N, 2, 2))
        cache, out = {}, ({} if by_assignment else np.zeros((N, 4, 4), complex))
        for assign in itertools.product(ACTIVE, repeat=n):
            acc = np.zeros((N, 4, 4), complex)
            for branch in itertools.product("LR", repeat=n):
                coeff, Lm, Rm, Lb, Rb = 1.0 + 0j, eye, eye, [], []
                for k in range(n - 1, -1, -1):          # rightmost Liouvillian acts first
                    a, S = assign[k], St[k][assign[k]]
                    if branch[k] == "L":
                        coeff *= -1j
                        Lm = S @ Lm
                        Lb.insert(0, (a, k))
                    else:
                        coeff *= 1j
                        Rm = Rm @ S
                        Rb.append((a, k))
                order = tuple(Rb + Lb)                  # Tr_B[L-ops rho_B R-ops] = <R-ops L-ops>
                if order not in cache:
                    cache[order] = self.bath.moment(tuple(o[0] for o in order), tuple(o[1] for o in order), C, Tt, N)
                acc += (coeff * cache[order])[:, None, None] * kron_RL(Rm, Lm)
            if by_assignment:
                out[assign] = acc
            else:
                out += acc
        return out

    # -------------------------------------------------------------------------------------
    def K(self, t: float, nq: int, h: float | None, orders=(2, 3, 4), resolve: bool = False, chunk: int = 3000):
        """K_n(t) for n in orders.  resolve=False: {n: (4,4)};  resolve=True: {n: (n+1,4,4)} split by
        the number nX of linear (AA) vertices (classes sum to the total; K4 cumulant subtraction resolved too)."""
        res = {}
        for order in orders:
            T, W = nested_nodes(t, order - 1, nq, h)
            acc = np.zeros((order + 1, 4, 4), complex) if resolve else np.zeros((4, 4), complex)
            for s in range(0, len(W), chunk):
                Tc, Wc = T[s:s + chunk], W[s:s + chunk]
                if not resolve:
                    M = self.liouville_moment(Tc)
                    if order == 4:
                        P = {pq: self.liouville_moment(Tc[:, list(pq)]) for pq in PAIRS_K4}
                        M = M - P[(0, 1)] @ P[(2, 3)] - P[(0, 2)] @ P[(1, 3)] - P[(0, 3)] @ P[(1, 2)]
                    acc += np.einsum("n,nij->ij", Wc, M)
                else:
                    Ma = self.liouville_moment(Tc, by_assignment=True)
                    if order == 4:
                        P = {pq: self.liouville_moment(Tc[:, list(pq)], by_assignment=True) for pq in PAIRS_K4}
                    for a, M in Ma.items():
                        if order == 4:
                            M = (M - P[(0, 1)][a[0], a[1]] @ P[(2, 3)][a[2], a[3]]
                                 - P[(0, 2)][a[0], a[2]] @ P[(1, 3)][a[1], a[3]]
                                 - P[(0, 3)][a[0], a[3]] @ P[(1, 2)][a[1], a[2]])
                        acc[sum(1 for x in a if x == 1)] += np.einsum("n,nij->ij", Wc, M)
            res[order] = acc
        return res


# ---------------------------------------------------------------------------------------------
# K at sparse nodes
# ---------------------------------------------------------------------------------------------
@dataclass
class KNodes:
    """K_n at nodes t_k = k node_dt (K_n(0) = 0).  K[n]: (nt,4,4), or (n+1, nt,4,4) if resolved by nX."""
    t: np.ndarray
    K: dict = field(default_factory=dict)
    resolved: bool = False

    def total(self, n: int) -> np.ndarray:
        return self.K[n].sum(0) if self.resolved else self.K[n]


def node_times(tmax: float, node_dt: float) -> np.ndarray:
    return np.arange(0.0, tmax + 1e-9, node_dt)


def K_nodes(frame: PolaronFrame, tmax: float, node_dt: float, nq: int, panel_h: float | None,
            orders=(2, 3, 4), resolve: bool = False, engine: MomentEngine | None = None) -> KNodes:
    eng = engine or MomentEngine(frame)
    ts = node_times(tmax, node_dt)
    K = {o: np.zeros(((o + 1,) if resolve else ()) + (len(ts), 4, 4), complex) for o in orders}
    for i, t in enumerate(ts):
        if t == 0:
            continue
        r = eng.K(float(t), nq, panel_h, orders, resolve)
        for o in orders:
            if resolve:
                K[o][:, i] = r[o]
            else:
                K[o][i] = r[o]
    return KNodes(t=ts, K=K, resolved=resolve)


def rms_norm(K: np.ndarray) -> float:
    """sqrt(mean_k ||K(t_k)||_F^2) over k >= 1 (the node t=0 is excluded)."""
    return float(np.sqrt(np.mean(np.linalg.norm(K[1:], axis=(1, 2)) ** 2)))


def generator_ratio(kn: KNodes, n: int) -> float:
    """r_n = rms ||K_n|| / rms ||K_2||  (r3, r4)."""
    return rms_norm(kn.total(n)) / rms_norm(kn.total(2))
