import numpy as np
import pytest

from pricers import bs, fd

REF_CALL = bs.price(100.0, 100.0, 1.0, 0.2, "C", 0.05)


def _cn(n, right="C", **kw):
    return fd.price(100.0, 100.0, 1.0, 0.2, right, 0.05, n_x=n, scheme="cn", **kw).price


def test_crank_nicolson_error_ratio_on_grid_doubling_is_second_order():
    """Strike on a node, Rannacher start: err(n)/err(2n) in [3.3, 4.7] (oracle: QuantLib's -> 4.0)."""
    errs = [abs(_cn(n) - REF_CALL) for n in (50, 100, 200, 400, 800)]
    for e1, e2 in zip(errs[:-1], errs[1:], strict=True):
        assert 3.3 <= e1 / e2 <= 4.7
    assert errs[-1] < 2e-4


def test_crank_nicolson_stays_second_order_with_spot_between_nodes():
    """S = 100.37319 (ln(S/K) = 0.37% of a step at n_x = 200): the spot is read off the spline, h = half / m
    exactly, and err(n)/err(2n) is 4.00 / 4.00 / 4.00 for n = 200..1600 (measured); an h nudged to put S
    on a node stalled at ratio 1.00 between n = 400 and 800 here."""
    S = 100.37319
    ref = bs.price(S, 100.0, 1.0, 0.2, "C", 0.05, 0.01)
    errs = [abs(fd.price(S, 100.0, 1.0, 0.2, "C", 0.05, 0.01, n_x=n).price - ref) for n in (200, 400, 800, 1600)]
    for e1, e2 in zip(errs[:-1], errs[1:], strict=True):
        assert 3.3 <= e1 / e2 <= 4.7, errs
    assert errs[2] < 2e-4
    assert fd.make_grid(S, 100.0, 1.0, 0.2, 800).h == pytest.approx((1.0 + abs(np.log(S / 100.0))) / 400, rel=1e-12)


def test_implicit_scheme_is_first_order_and_explicit_is_stable_at_its_own_dt():
    e_impl = [abs(fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, n_x=n, scheme="implicit").price - REF_CALL)
              for n in (100, 200, 400)]
    assert 1.7 <= e_impl[0] / e_impl[1] <= 2.9 and 1.7 <= e_impl[1] / e_impl[2] <= 2.9
    res = fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, n_x=200, scheme="explicit")
    lam_min = 1.0 * 0.2**2 / fd.make_grid(100.0, 100.0, 1.0, 0.2, 200).h**2
    assert res.n_t == int(np.ceil(lam_min)) + 1                 # the smallest stable count plus one
    assert abs(res.price - REF_CALL) < 1e-3
    at_limit = fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, n_x=200, n_t=int(np.ceil(lam_min)), scheme="explicit").price
    assert 1e-3 < abs(at_limit - REF_CALL) < 2e-3              # exactly at the limit the top mode is undamped
    with pytest.raises(ValueError):
        fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, n_x=200, n_t=20, scheme="explicit")


def test_european_fd_agrees_with_black_scholes_off_the_strike_node_and_for_puts():
    for S, right in ((90.0, "C"), (95.3, "C"), (100.0, "P"), (117.2, "P")):
        ref = bs.price(S, 100.0, 1.0, 0.2, right, 0.05, 0.01)
        px = fd.price(S, 100.0, 1.0, 0.2, right, 0.05, 0.01, n_x=800, scheme="cn").price
        assert px == pytest.approx(ref, abs=2e-4)


def test_put_call_parity_of_fd_prices_to_1e4():
    c, p = _cn(400), _cn(400, "P")
    assert c - p == pytest.approx(100.0 - 100.0 * np.exp(-0.05), abs=1e-4)


def test_brennan_schwartz_equals_psor_on_the_same_grid_to_1e6(oracle):
    """Hull's put (S=K=50, r=10%, sigma=40%, T=5/12) on one 150 x 150 grid: the two LCP solvers agree to 1e-6."""
    o = oracle["hull_american_put_tree"]
    kw = dict(n_x=150, exercise="american")
    a = fd.price(o["S"], o["K"], o["T"], o["sigma"], "P", o["r"], american="bs", **kw)
    b = fd.price(o["S"], o["K"], o["T"], o["sigma"], "P", o["r"], american="psor", **kw)
    assert a.price == pytest.approx(b.price, abs=1e-6)
    assert b.psor_iterations > 0 and a.psor_iterations == 0


def test_hull_american_put_converges_to_4_2842_with_brennan_schwartz(oracle):
    """Hull's put on an 800 x 800 grid vs the converged 4.2842 (QuantLib QdFpAmericanEngine), 1e-3."""
    o = oracle["hull_american_put_tree"]
    res = fd.price(o["S"], o["K"], o["T"], o["sigma"], "P", o["r"], n_x=800, exercise="american", american="bs")
    assert res.price == pytest.approx(o["converged"], abs=1e-3)
    assert res.american_solver == "bs" and res.exercise == "american"


def test_lsm_table1_continuous_american_column_to_1e3_by_brennan_schwartz(oracle):
    """All 20 Longstaff-Schwartz Table 1 cases (K=40, r=6%) vs the continuous-American reference."""
    o = oracle["american_put_lsm2001_table1"]
    tol = o["tolerances"]["american_continuous_vs_converged_psor"]
    for row in o["rows"]:
        px = fd.price(row["S"], o["K"], row["T"], row["sigma"], "P", o["r"], n_x=800, exercise="american", american="bs").price
        assert px == pytest.approx(row["american_continuous"], abs=tol), row
        assert px >= row["european_bs"] - 1e-3 and px >= max(o["K"] - row["S"], 0.0)


def test_psor_reaches_the_continuous_reference_within_5e3_on_three_cases(oracle):
    """PSOR is the pure-Python reference solver: a coarser grid, so a looser bar."""
    o = oracle["american_put_lsm2001_table1"]
    for row in (o["rows"][0], o["rows"][10], o["rows"][19]):
        px = fd.price(row["S"], o["K"], row["T"], row["sigma"], "P", o["r"], n_x=200, exercise="american", american="psor").price
        assert px == pytest.approx(row["american_continuous"], abs=5e-3), row


def test_american_call_needs_psor_and_american_dominates_european():
    with pytest.raises(ValueError):
        fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, 0.03, n_x=100, exercise="american", american="bs")
    am = fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, 0.03, n_x=100, exercise="american", american="psor").price
    eu = fd.price(100.0, 100.0, 1.0, 0.2, "C", 0.05, 0.03, n_x=100).price
    assert am >= eu - 1e-10


def test_bumped_greeks_match_black_scholes_for_the_european_case():
    g = fd.greeks(100.0, 100.0, 1.0, 0.2, "C", 0.05, n_x=800)
    ref = bs.greeks(100.0, 100.0, 1.0, 0.2, "C", 0.05)
    assert g.price == pytest.approx(ref.price, abs=2e-4)
    assert g.delta == pytest.approx(ref.delta, abs=2e-4)
    assert g.gamma == pytest.approx(ref.gamma, rel=1e-2)
    assert g.vega == pytest.approx(ref.vega, rel=1e-3)
    assert g.rho == pytest.approx(ref.rho, rel=1e-3)
    assert g.theta == pytest.approx(ref.theta, rel=3e-2)


def test_grid_puts_strike_on_the_centre_node_with_the_exact_step():
    g = fd.make_grid(90.0, 100.0, 1.0, 0.2, n_x=200)
    assert np.isclose(np.exp(g.x[g.i_strike]), 100.0) and g.i_strike == 100
    assert g.h == pytest.approx((5.0 * 0.2 + abs(np.log(0.9))) / 100, rel=1e-12)
    assert len(g.x) == 201 and g.n_t == 200
    assert g.x[0] <= np.log(90.0) - 5.0 * 0.2 + 1e-12 and g.x[-1] >= np.log(90.0) + 5.0 * 0.2 - 1e-12


def test_cn_matches_quantlib_fd_engine_within_their_combined_error(ql, ql_bs_process, ql_option):
    """Different meshers, same PDE: both within 1e-3 of the closed form at n = 400 and of each other."""
    proc = ql_bs_process(100.0, 0.05, 0.0, 0.2)
    opt = ql_option("C", 100.0, 365)
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 400, 400))
    ours = _cn(400)
    assert abs(ours - REF_CALL) < 1e-3 and abs(opt.NPV() - REF_CALL) < 1e-3
    assert ours == pytest.approx(opt.NPV(), abs=2e-3)


def test_american_put_with_dividend_yield_matches_quantlib_qdfp_to_1e3(ql, ql_bs_process, ql_option):
    """A case outside the fixture (S=100, K=110, sigma=30%, T=0.75, r=4%, q=2%): Brennan-Schwartz on
    800 x 800 vs QuantLib's QdFpAmericanEngine (high precision)."""
    proc = ql_bs_process(100.0, 0.04, 0.02, 0.3)
    opt = ql_option("P", 110.0, 274, american=True)
    opt.setPricingEngine(ql.QdFpAmericanEngine(proc, ql.QdFpAmericanEngine.highPrecisionScheme()))
    T = 274 / 365
    ours = fd.price(100.0, 110.0, T, 0.3, "P", 0.04, 0.02, n_x=800, exercise="american", american="bs").price
    assert ours == pytest.approx(opt.NPV(), abs=1e-3)
    assert ours > bs.price(100.0, 110.0, T, 0.3, "P", 0.04, 0.02)
