r"""Surrogate model and acquisition for the frame grid.

Features (normalised to [0,1] over the grid):  F0 = (log alpha, log omega_h);  F3 = F0 + (B, lambda_p, rms||K2||).
Model: PCA of Y to nc <= 6 components; for each component k a GP  C * Matern(nu=5/2, ARD) + White  on F, fitted to the
component scores / std.  Prediction: inverse PCA of the GP means; uncertainty u = sqrt(sum_k (std_k * scale_k)^2).
Acquisition (start: n_init random frames): random | maximin (largest min distance in F0) | gp_u (largest u) |
hybrid (u_norm + lam * D_norm).  Accuracy at the checkpoints CH: median / 90th percentile of ||Yhat - Y|| / ||Y|| and the
aggregate ||Yhat - Y||_F / ||Y||_F over the not-acquired frames.
"""
from __future__ import annotations

import warnings

import numpy as np

CH = [10, 15, 20, 25, 30, 40]


def nz(M):
    return (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-12)


def feats(d, kind):
    X = d["X"]; F = d["F"]; U0 = nz(np.c_[np.log(X[:, 0]), np.log(X[:, 1])])
    return U0 if kind == "F0" else nz(np.c_[U0, F[:, 0], F[:, 1], F[:, 2]])


def fit(Y, Xf, tr, nc=6):
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    warnings.filterwarnings("ignore")
    nc = min(nc, len(tr) - 1); p = PCA(nc).fit(Y[tr]); S = p.transform(Y[tr]); gps = []; sc = []
    d = Xf.shape[1]
    for k in range(nc):
        s = S[:, k].std() + 1e-12
        gp = GPR(C(1.0) * Matern([0.4] * d, nu=2.5, length_scale_bounds=(0.05, 10)) + WhiteKernel(1e-3, (1e-8, 1e-1)), n_restarts_optimizer=0).fit(Xf[tr], S[:, k] / s)
        gps.append(gp); sc.append(s)
    return p, gps, np.array(sc)


def predict(m, Xf, idx, std=False):
    p, gps, sc = m; mu = np.zeros((len(idx), len(gps))); var = np.zeros(len(idx))
    for k, gp in enumerate(gps):
        if std:
            a, s = gp.predict(Xf[idx], return_std=True); mu[:, k] = a * sc[k]; var += (s * sc[k]) ** 2
        else:
            mu[:, k] = gp.predict(Xf[idx]) * sc[k]
    Yh = p.inverse_transform(mu)
    return (Yh, np.sqrt(var)) if std else Yh


def _next(strategy, rng, cand, D, Y, Xf, acq, lam):
    if strategy == "random":
        return rng.integers(len(cand))
    if strategy == "maximin":
        return int(np.argmax(D))
    m = fit(Y, Xf, acq); _, u = predict(m, Xf, cand, std=True); u = (u - u.min()) / (u.max() - u.min() + 1e-12)
    if strategy == "gp_u":
        return int(np.argmax(u))
    Dn = (D - D.min()) / (D.max() - D.min() + 1e-12)
    return int(np.argmax(u + lam * Dn))


def run(d, strategy, kind, seed, n_init=6, lam=0.5):
    """Generator-level learning curve {n_acquired: {med, p90, agg}} for one case."""
    Y = d["Y"]; N = len(Y); Xf = feats(d, kind); U0 = feats(d, "F0"); rng = np.random.default_rng(seed)
    perm = rng.permutation(N); acq = list(perm[:n_init]); rest = list(perm[n_init:]); res = {}
    nrm = np.linalg.norm(Y, axis=1)
    while True:
        n = len(acq)
        if n in CH:
            m = fit(Y, Xf, acq); te = np.array(rest); Yh = predict(m, Xf, te); e = np.linalg.norm(Yh - Y[te], axis=1)
            res[n] = dict(med=float(np.median(e / nrm[te])), p90=float(np.percentile(e / nrm[te], 90)), agg=float(np.linalg.norm(Yh - Y[te]) / np.linalg.norm(Y[te])))
        if n >= max(CH):
            break
        cand = np.array(rest)
        D = np.min(np.linalg.norm(U0[cand][:, None, :] - U0[acq][None, :, :], axis=2), axis=1)
        j = _next(strategy, rng, cand, D, Y, Xf, acq, lam)
        acq.append(cand[j]); rest.remove(cand[j])
    return res


def acquire(d, strategy, kind, seed, nmax, n_init=6, lam=0.5):
    """Indices of the first nmax frames chosen by a strategy (same rules as run)."""
    Y = d["Y"]; N = len(Y); Xf = feats(d, kind); U0 = feats(d, "F0"); rng = np.random.default_rng(seed)
    perm = rng.permutation(N); acq = list(perm[:n_init]); rest = list(perm[n_init:])
    while len(acq) < nmax:
        cand = np.array(rest); D = np.min(np.linalg.norm(U0[cand][:, None, :] - U0[acq][None, :, :], axis=2), axis=1)
        j = _next(strategy, rng, cand, D, Y, Xf, acq, lam)
        acq.append(cand[j]); rest.remove(cand[j])
    return acq
