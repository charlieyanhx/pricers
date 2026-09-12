import numpy as np
import pytest

from pricers import bs, heston


def _fo_params(oracle):
    o = oracle["heston"]["fang_oosterlee_2008"]
    return o, heston.HestonParams(o["v0"], o["kappa"], o["theta"], o["sigma_v"], o["rho"])


def test_fang_oosterlee_heston_calls_match_published_values_to_1e6(oracle):
    """Fang & Oosterlee 2008 eq.(53), Tables 4-5, Feller violated; defaults N=256, L=12, numeric c4."""
    o, p = _fo_params(oracle)
    assert not p.feller() and o["feller_violated"]
    for key, T in (("call_T1", 1.0), ("call_T10", 10.0)):
        px = heston.price(o["S"], o["K"], T, "C", p, o["r"], o["q"])
        assert px == pytest.approx(o[key]["published"], abs=o[key]["tol"]), key
        assert px == pytest.approx(o[key]["quantlib_5_engines"], abs=o[key]["tol"]), key


def test_plain_c2_range_saturates_at_4e5_for_t1_and_the_c4_term_fixes_it(oracle):
    """Measured: with c4=0 the T=1 error is -3.9e-5 for any N >= 128 (QuantLib's COSHestonEngine at
    L=12, N=256 gives the same -3.9e-5); with the c4 term it is 4.5e-8."""
    o, p = _fo_params(oracle)
    ref = o["call_T1"]["published"]
    for N in (256, 1024):
        assert heston.price(100.0, 100.0, 1.0, "C", p, N=N, c4=0) - ref == pytest.approx(-3.9e-5, abs=0.3e-5)
    assert abs(heston.price(100.0, 100.0, 1.0, "C", p, N=256) - ref) < 1e-7


def test_andersen_case1_needs_a_wide_plain_range_and_2048_terms(oracle):
    """sigma_v = 1, kappa = 0.5, T = 10: the numeric c4 explodes; c4=0 with L=30, N=2048 gives 1e-6."""
    o = oracle["heston"]["andersen_2008_case1"]
    p = heston.HestonParams(o["v0"], o["kappa"], o["theta"], o["sigma_v"], o["rho"])
    px = heston.price(o["S"], o["K"], o["T"], "C", p, o["r"], o["q"], N=2048, L=30.0, c4=0)
    assert px == pytest.approx(o["call"], abs=o["tol"])


def test_c2_formula_matches_numerical_second_derivative_of_log_cf(oracle):
    o, p = _fo_params(oracle)
    for T in (1.0, 10.0):
        c1, c2, _ = heston.cumulants_heston(T, p, 0.0, 0.0, c4=0)
        h = 1e-3

        def lf(u, T=T):
            return np.log(heston.cf_heston(np.array([u], dtype=complex), T, p, 0.0, 0.0))[0]

        assert c1 == pytest.approx(((lf(h) - lf(-h)) / (2j * h)).real, abs=1e-6)
        assert c2 == pytest.approx((-(lf(h) - 2 * lf(0.0) + lf(-h)) / h**2).real, abs=1e-6)


def test_cos_under_gbm_reproduces_black_scholes_and_the_paper_table2_to_1e8(oracle):
    o = oracle["bs_european"]["fang_oosterlee_gbm"]
    for K, ref in o["calls"].items():
        assert heston.price_gbm(o["S"], float(K), o["T"], o["sigma"], "C", o["r"]) == pytest.approx(ref, abs=o["tol"])
    for right in ("C", "P"):
        for S in (80.0, 100.0, 125.0):
            assert heston.price_gbm(S, 100.0, 1.0, 0.2, right, 0.05, 0.01) == pytest.approx(
                bs.price(S, 100.0, 1.0, 0.2, right, 0.05, 0.01), abs=1e-8)


def test_heston_put_call_parity_holds_by_construction(oracle):
    o, p = _fo_params(oracle)
    for T in (0.25, 1.0, 5.0):
        c = heston.price(100.0, 90.0, T, "C", p, 0.03, 0.01)
        pt = heston.price(100.0, 90.0, T, "P", p, 0.03, 0.01)
        assert c - pt == pytest.approx(100.0 * np.exp(-0.01 * T) - 90.0 * np.exp(-0.03 * T), abs=1e-10)


def test_heston_prices_are_monotone_in_strike_and_bounded_by_intrinsic(oracle):
    o, p = _fo_params(oracle)
    calls = [heston.price(100.0, K, 1.0, "C", p) for K in (60.0, 80.0, 100.0, 120.0, 140.0)]
    assert all(a > b for a, b in zip(calls[:-1], calls[1:], strict=True))
    assert calls[0] > 40.0 and calls[-1] > 0.0


def test_heston_matches_quantlib_cos_and_analytic_engines(ql, oracle):
    """QuantLib COSHestonEngine at L=16, N=256 and AnalyticHestonEngine, both to 2e-6 of ours at T=1 and T=10."""
    o, p = _fo_params(oracle)
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    proc = ql.HestonProcess(flat, flat, ql.QuoteHandle(ql.SimpleQuote(100.0)), p.v0, p.kappa, p.theta, p.sigma_v, p.rho)
    model = ql.HestonModel(proc)
    for T in (1, 10):
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(today + 365 * T))
        ours = heston.price(100.0, 100.0, float(T), "C", p)
        opt.setPricingEngine(ql.COSHestonEngine(model, 16, 256))
        assert ours == pytest.approx(opt.NPV(), abs=2e-6)
        opt.setPricingEngine(ql.AnalyticHestonEngine(model))
        assert ours == pytest.approx(opt.NPV(), abs=2e-6)


def test_heston_cos_with_nonzero_rate_and_dividend_matches_quantlib_analytic():
    """r=3%, q=1%, v0=0.04, kappa=2, theta=0.04, sigma_v=0.3, rho=-0.7, S=K=100: calls checked against
    QuantLib 1.43 AnalyticHestonEngine (1e-12, 10000) and COSHestonEngine(25, 1000) with exact Act/360
    maturities; both engines agree to 1e-9. Exercises the discount and dividend terms the r=q=0 fixture
    cannot; the put is checked through parity."""
    import math

    from pricers import heston

    p = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    for T, call_ref in [(0.5, 5.956581410), (1.0, 8.599588102), (3.0, 15.694072994)]:
        c = heston.price(100.0, 100.0, T, "C", p, r=0.03, q=0.01)
        put = heston.price(100.0, 100.0, T, "P", p, r=0.03, q=0.01)
        assert abs(c - call_ref) < 1e-6, (T, c)
        assert abs((c - put) - (100.0 * math.exp(-0.01 * T) - 100.0 * math.exp(-0.03 * T))) < 1e-9
