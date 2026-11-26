"""Regression + invariant tests for goa_pqme.

Run:  python tests/test_goa_pqme.py          (or: python -m pytest tests/test_goa_pqme.py)
      python tests/test_goa_pqme.py --fast   (skip the published-frame reproduction)

Comparisons with the original engine run only if it is available in legacy/src/ (not part of this repository);
the published-frame test needs the TEMPO reference data/trajectories/G0p1_OD0_Binf_y0p75.csv.  Missing inputs -> skipped.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("GOA_WORK", tempfile.mkdtemp(prefix="goa_test_"))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "legacy", "src"))
sys.dont_write_bytecode = True

import numpy as np

import goa_pqme as gp
from goa_pqme.liouville import TRACE_ROW, commutator_generator, to_interaction

P = gp.ModelParams(Gamma=0.2, omega_DA=0.28125, beta=5.0, y0=0.75)
A, W = 2.0, 0.29


class SkipTest(Exception):
    pass


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise SkipTest(msg)


def _legacy():
    if not os.path.isdir(os.path.join(ROOT, "legacy", "src")):
        _skip("original engine (legacy/src) not available")
    import comp
    import fastk
    import goa_engine as ge
    import goa_user as G
    return G, ge, comp, fastk


def _close(a, b, tol, what):
    a, b = np.asarray(a), np.asarray(b)
    err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)
    assert err < tol, "%s: relative difference %.3e >= %.1e" % (what, err, tol)
    return err


def test_normal_modes_and_Er():
    G, *_ = _legacy()
    m = gp.normal_modes(P)
    om, R, Er = G.goa_normal_modes(dict(N=50, eta_secondary=1.0, omega_c=1.0, Omega=0.5, y0=0.75))
    _close(m.omega, om, 1e-14, "omega"); _close(m.R, R, 1e-12, "R")
    assert abs(m.Er - P.Er) < 1e-12 and abs(m.Er - Er) < 1e-14


def test_frame_and_bath_functions():
    G, *_ = _legacy()
    fr = gp.make_frame(P, A, W)
    t = np.linspace(0, 10, 201)
    old = G.bath_functions(fr.omega, fr.modes.R, P.beta, A, W, t)
    for k in ("phi", "mixed", "linear"):
        _close(getattr(fr, k)(t), old[k], 1e-13, k)
    assert abs(fr.B - old["B"]) < 1e-15 and abs(fr.lambda_p - old["lambda_p"]) < 1e-15
    assert np.allclose(fr.U @ np.diag(fr.E) @ fr.U.conj().T, fr.H_tilde)


def test_tcl2_matches_legacy():
    G, ge, *_ = _legacy()
    t = np.arange(0, 6 + 1e-9, 0.05)
    g = ge.build(ge.case_params(P.Gamma, P.omega_DA, P.beta, P.y0), A, W, t)
    tc = gp.TCL2(gp.make_frame(P, A, W), t)
    _close(tc.C, g["cp"], 1e-13, "C_ab"); _close(tc.Cm, g["cm"], 1e-13, "C^-_ab")
    _close(tc.L, ge.generator(g), 1e-12, "L(t)")
    assert np.max(np.abs(np.einsum("i,tij->tj", TRACE_ROW, tc.L))) < 1e-12      # trace preserving


def test_K_nodes_match_fastk():
    G, ge, comp, fastk = _legacy()
    eng, *_ = comp.setup(A, W, GAM=P.Gamma, OD=P.omega_DA, BETA=P.beta, Y0=P.y0)
    f = fastk.FastTCL(eng.bath, eng.E, eng.S)
    kn = gp.K_nodes(gp.make_frame(P, A, W), 3.0, 1.5, 3, 3.0)
    for i, t in enumerate(kn.t[1:], 1):
        old = fastk.compute_K(f, float(t), 3, 3.0)
        for n in (2, 3, 4):
            _close(kn.K[n][i], old[n], 1e-10, "K%d(t=%g)" % (n, t))


def test_channel_resolution_sums_to_total():
    fr = gp.make_frame(P, A, W)
    eng = gp.MomentEngine(fr)
    tot, res = eng.K(2.0, 3, 2.0), eng.K(2.0, 3, 2.0, resolve=True)
    for n in (2, 3, 4):
        _close(res[n].sum(0), tot[n], 1e-12, "sum of nX classes of K%d" % n)


def test_K2_equals_tcl2_dissipator():
    """K2(t) = P^-1 (L(t) - L_H) P : the higher-order engine and the TCL2 code are the same theory."""
    fr = gp.make_frame(P, A, W)
    t = np.arange(0, 3 + 1e-9, 0.005)
    L = gp.TCL2(fr, t).L
    eng = gp.MomentEngine(fr)
    for tt in (1.5, 3.0):
        i = int(round(tt / 0.005))
        DI = to_interaction((L[i] - commutator_generator(fr.E))[None], fr.E, [tt])[0]
        _close(eng.K(tt, 8, 0.5, orders=(2,))[2], DI, 1e-5, "K2 vs TCL2 dissipator at t=%g" % tt)


def test_trace_preservation_K34():
    eng = gp.MomentEngine(gp.make_frame(P, A, W))
    for n, K in eng.K(3.0, 3, 3.0).items():
        assert np.max(np.abs(TRACE_ROW @ K)) < 1e-14, "K%d not trace preserving" % n


def test_two_point_moment_is_tcl2_tensor():
    fr = gp.make_frame(P, A, W)
    bath = gp.ResidualBath(fr)
    C, _ = gp.correlation_tensors(fr, [1.3])
    for a in (1, 2, 3):
        for b in (1, 2, 3):
            m = bath.moment_at((a, b), (1.3, 0.0))
            assert abs(m - C[a, b, 0]) < 1e-12 * max(1, abs(C[a, b, 0])), (a, b, m, C[a, b, 0])


def test_coherence_readout_matches_legacy():
    G, ge, *_ = _legacy()
    t = np.arange(0, 4 + 1e-9, 0.05)
    g = ge.build(ge.case_params(P.Gamma, P.omega_DA, P.beta, P.y0), A, W, t)
    fr = gp.make_frame(P, A, W); tc = gp.TCL2(fr, t)
    rho = gp.propagate(tc.L, t, gp.rho0_donor(fr))
    _close(rho, ge.propagate(g, ge.generator(g)), 1e-12, "rho(t)")
    from goa_pqme.dynamics import q1
    _close(q1(fr, t, tc.C, tc.Cm, rho), ge.q1_conv(g, rho, ge.q1_kernel(g, a=3)), 1e-12, "Q1")


def test_io_roundtrip_and_al_consistency():
    """save_frame/load_frame round trip; the AL site-basis dK fed back through pop_curves reproduces TCL234."""
    import tempfile as tf
    from goa_pqme.al.data import frame_features
    from goa_pqme.al.pop_eval import pop_curves
    from goa_pqme.io import frame_filename, load_frame, save_frame
    num = gp.Numerics(node_dt=1.5, nq=3, panel_h=3.0, dt=0.05, tmax=4.5)
    res = gp.run_frame(P, A, W, num, coherence=True)
    d = tf.mkdtemp(prefix="goa_io_")
    path = frame_filename(d, "TEST", A, W)
    save_frame(path, res, P, num)
    z = load_frame(path)
    for m in res.dP:
        assert np.array_equal(z["dP"][m], res.dP[m])
    assert z["params"]["beta"] == P.beta and z["numerics"]["panel_h"] == 3.0 and set(z["coherence"]) == set(res.coherence)
    y, F, tn = frame_features(z)
    n = len(y) // 2
    _, dP = pop_curves(P, A, W, y[:n] + 1j * y[n:], tn)
    _close(dP, res.dP["TCL234"], 1e-10, "pop_curves(true dK) vs TCL234")


def test_published_frame():
    """README / results_full.jsonl: G0p1_OD0_Binf_y0p75, alpha=0.5, omega_h=0.05, production numerics."""
    cid = "G0p1_OD0_Binf_y0p75"
    try:
        ref = gp.load_reference(cid)
    except FileNotFoundError:
        _skip("no TEMPO reference in data/trajectories")
    res = gp.run_frame(gp.get_case(cid), 0.5, 0.05, gp.PRODUCTION)
    err = {m: gp.half_rmse(res.t, y, ref.t.values, ref.dP.values) for m, y in res.dP.items()}
    published = dict(TCL2=0.011852402820054998, K3=0.049987551088432294, K4=0.024535302560152697, TCL234=0.07185910032995305)
    for m, v in published.items():
        assert abs(err[m] - v) < 1e-9 * v, (m, err[m], v)
    assert abs(res.r3 - 0.3367006298392812) < 1e-9 and abs(res.r4 - 0.34388540200210643) < 1e-9


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for name, fn in tests:
        if fast and name == "test_published_frame":
            continue
        try:
            fn(); print("PASS", name)
        except BaseException as e:  # noqa
            if type(e).__name__ in ("SkipTest", "Skipped"):
                skipped += 1; print("SKIP", name, "->", e)
            elif isinstance(e, Exception):
                failed += 1; print("FAIL", name, "->", repr(e))
            else:
                raise
    print("%d tests, %d failed, %d skipped" % (len(tests) - (1 if fast else 0), failed, skipped))
    sys.exit(1 if failed else 0)
