import numpy as np
import pytest

from pricers import bs, heston


def _fo_params(oracle):
    o = oracle["heston"]["fang_oosterlee_2008"]
    return o, heston.HestonParams(o["v0"], o["kappa"], o["theta"], o["sigma_v"], o["rho"])


def test_fang_oosterlee_heston_calls_match_published_values_to_1e6(oracle):
    """Fang & Oosterlee 2008 eq.(53), Tables 4-5, Feller violated; defaults N=1024, L=12, numeric c4."""
    o, p = _fo_params(oracle)
    assert not p.feller() and o["feller_violated"]
    for key, T in (("call_T1", 1.0), ("call_T10", 10.0)):
        px = heston.price(o["S"], o["K"], T, "C", p, o["r"], o["q"])
        assert px == pytest.approx(o[key]["published"], abs=o[key]["tol"]), key
        assert px == pytest.approx(o[key]["quantlib_5_engines"], abs=o[key]["tol"]), key


def test_plain_c2_range_saturates_at_4e5_for_t1_and_the_c4_term_fixes_it(oracle):
    """Measured: with c4=0 the T=1 error is -3.9e-5 for any N >= 128 (QuantLib's COSHestonEngine at
    L=12, N=256 gives the same -3.9e-5); with the c4 term it is 4.5e-8 at N=256 (Fang-Oosterlee call)."""
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


def test_c2_formula_matches_numerical_second_derivative_of_log_cf_to_1e9(oracle):
    """Closed-form c1, c2 vs Richardson-extrapolated central differences of log phi at u = 0 (h = 5e-3
    and h/2; the plain h = 1e-3 stencil only reaches 5e-8 at T = 10): both to 1e-9 at T = 1 and T = 10,
    Fang-Oosterlee parameters."""
    o, p = _fo_params(oracle)
    for T in (1.0, 10.0):
        c1, c2, _ = heston.cumulants_heston(T, p, 0.0, 0.0, c4=0)

        def lf(u, T=T):
            return np.log(heston.cf_heston(np.array([u], dtype=complex), T, p, 0.0, 0.0))[0]

        def d1(h):
            return ((lf(h) - lf(-h)) / (2j * h)).real

        def d2(h):
            return (-(lf(h) - 2 * lf(0.0) + lf(-h)) / h**2).real

        h = 5e-3
        assert c1 == pytest.approx((4 * d1(h / 2) - d1(h)) / 3, abs=1e-9)
        assert c2 == pytest.approx((4 * d2(h / 2) - d2(h)) / 3, abs=1e-9)


def _c2_paper_as_printed(T: float, p: heston.HestonParams) -> float:
    """Fang & Oosterlee (2008) Table 11, c2 for Heston, transcribed as printed."""
    lam, eta, ub, u0, rho = p.kappa, p.sigma_v, p.theta, p.v0, p.rho
    e = np.exp(-lam * T)
    return (1 / (8 * lam**3)) * (
        eta * T * lam * e * (u0 - ub) * (8 * lam * rho - 4 * eta)
        + lam * rho * eta * (1 - e) * (16 * ub - 8 * u0)
        + 2 * ub * lam * T * (-4 * lam * rho * eta + eta**2 + 4 * lam**2)
        + eta**2 * ((ub - 2 * u0) * np.exp(-2 * lam * T) + ub * (6 * e - 7) + 2 * u0)
        + 8 * lam**2 * (u0 - ub) * (1 - e))


def test_paper_table11_c2_as_printed_is_short_by_eta2_theta_term(oracle):
    """The printed Table 11 c2 is below the variance of ln(S_T/S_0) by exactly eta^2 theta (1 - e^{-kappa T})
    / (4 kappa^3): 6.660e-4 at T = 1 (2.1% of c2) and 8.394e-4 at T = 10, to 1e-15; the previous test
    shows the closed form here is the one the characteristic function agrees with."""
    o, p = _fo_params(oracle)
    for T, gap_expected in ((1.0, 6.659672e-4), (10.0, 8.394212e-4)):
        _, c2, _ = heston.cumulants_heston(T, p, 0.0, 0.0, c4=0)
        gap = c2 - _c2_paper_as_printed(T, p)
        assert gap == pytest.approx(p.sigma_v**2 * p.theta * (1 - np.exp(-p.kappa * T)) / (4 * p.kappa**3), abs=1e-15)
        assert gap == pytest.approx(gap_expected, abs=1e-9)


def test_cos_error_falls_exponentially_in_n(oracle):
    """T = 1 Fang-Oosterlee call, numeric c4, L = 12: |error| at N = 32 / 64 / 128 / 256 is about
    2.5e-1 / 6.2e-3 / 4.1e-4 / 4.5e-8, i.e. each doubling of N cuts the error by more than 10x
    (measured 40x, 15x, 9000x), and N = 256 is already at 1e-7."""
    o, p = _fo_params(oracle)
    ref = o["call_T1"]["published"]
    errs = [abs(heston.price(100.0, 100.0, 1.0, "C", p, N=N) - ref) for N in (32, 64, 128, 256)]
    for e_n, e_2n in zip(errs[:-1], errs[1:], strict=True):
        assert e_n / e_2n > 10.0, errs
    assert errs[0] > 1e-2 and errs[-1] < 1e-7


def test_cos_range_follows_moneyness_deep_in_and_out_of_the_money(oracle):
    """The truncation range sits at x + c1 with x = ln(S/K), so it holds at any moneyness: GBM at 6, 8 and
    10 standard deviations from the strike (T down to 0.02) reproduces Black-Scholes to 1e-10 (a range
    placed at c1 alone returned -99.8 for the K = 200, T = 0.02 call); Heston with the Fang-Oosterlee
    parameters, r = q = 0, at 6-38 sd matches QuantLib 1.43 AnalyticHestonEngine (Andersen-Piterbarg,
    Gauss-Laguerre 192) to 1e-8: K = 130 30-day call 9.00e-10, K = 80 30-day put 5.85395771e-4,
    K = 150 / 200 7-day calls and K = 50 / 60 7-day puts 0 (QuantLib's own values are within 6e-9 of 0)."""
    for sd in (6.0, 8.0, 10.0):
        for T in (0.02, 0.05, 0.5):
            for right in ("C", "P"):
                for sign in (1.0, -1.0):
                    K = 100.0 * np.exp(-sign * sd * 0.2 * np.sqrt(T))
                    assert heston.price_gbm(100.0, K, T, 0.2, right, 0.05) == pytest.approx(
                        bs.price(100.0, K, T, 0.2, right, 0.05), abs=1e-10), (sd, T, right, sign)
    o, p = _fo_params(oracle)
    for K, days, right, ref in ((130.0, 30, "C", 9.00e-10), (80.0, 30, "P", 5.85395771e-4), (150.0, 7, "C", 0.0),
                                (200.0, 7, "C", 0.0), (60.0, 7, "P", 0.0), (50.0, 7, "P", 0.0)):
        assert heston.price(o["S"], K, days / 365, right, p, o["r"], o["q"]) == pytest.approx(ref, abs=1e-8), (K, days)
        assert heston.price(o["S"], K, days / 365, right, p, o["r"], o["q"]) >= 0.0


def test_feller_violated_high_vol_of_vol_needs_1024_terms():
    """v0 = 0.089, kappa = 0.57, theta = 0.065, sigma_v = 0.93, rho = -0.78 (2 kappa theta = 0.074 <
    sigma_v^2 = 0.865), r = 2%, q = 1%: the c4 term widens the range to 13.5 half-width (5.5 without it)
    and 256 terms no longer resolve it. QuantLib 1.43 AnalyticHestonEngine (Andersen-Piterbarg,
    Gauss-Laguerre 192): put K = 100.6, T = 610/365: 9.122628532037; call K = 95, T = 400/365:
    12.486828602536. N = 256 is off by 2.2e-3 / 6.6e-4, the default N = 1024 by 3e-8 / 4e-9.
    On a 400-case random sweep of Feller-violated and satisfied parameters (|ln S/K| <= 0.7, 5-730 days)
    N = 256 exceeds 1e-6 in 30 cases (worst 8.6e-4); N = 1024 in none (worst 3e-9)."""
    p = heston.HestonParams(v0=0.089, kappa=0.57, theta=0.065, sigma_v=0.93, rho=-0.78)
    assert not p.feller()
    for K, T, right, ref in ((100.6, 610 / 365, "P", 9.122628532037), (95.0, 400 / 365, "C", 12.486828602536)):
        assert abs(heston.price(100.0, K, T, right, p, 0.02, 0.01, N=256) - ref) > 5e-4
        assert heston.price(100.0, K, T, right, p, 0.02, 0.01) == pytest.approx(ref, abs=1e-7)


def test_cos_under_gbm_reproduces_black_scholes_and_the_paper_table2_to_1e8(oracle):
    """Fang & Oosterlee 2008 Table 2 (GBM calls, S=100, r=10%, sigma=25%, T=0.1, K in {80,100,120}), then
    Black-Scholes with r=5%, q=1% at S in {80,100,125}, calls and puts; the machinery check."""
    o = oracle["bs_european"]["fang_oosterlee_gbm"]
    for K, ref in o["calls"].items():
        assert heston.price_gbm(o["S"], float(K), o["T"], o["sigma"], "C", o["r"]) == pytest.approx(ref, abs=o["tol"])
    for right in ("C", "P"):
        for S in (80.0, 100.0, 125.0):
            assert heston.price_gbm(S, 100.0, 1.0, 0.2, right, 0.05, 0.01) == pytest.approx(
                bs.price(S, 100.0, 1.0, 0.2, right, 0.05, 0.01), abs=1e-8)


def test_heston_put_call_parity_holds_by_construction(oracle):
    """Fang-Oosterlee parameters, r=3%, q=1%, K=90: the call is put + parity, so C - P is exact to 1e-10."""
    o, p = _fo_params(oracle)
    for T in (0.25, 1.0, 5.0):
        c = heston.price(100.0, 90.0, T, "C", p, 0.03, 0.01)
        pt = heston.price(100.0, 90.0, T, "P", p, 0.03, 0.01)
        assert c - pt == pytest.approx(100.0 * np.exp(-0.01 * T) - 90.0 * np.exp(-0.03 * T), abs=1e-10)


def test_heston_prices_are_monotone_in_strike_and_bounded_by_intrinsic(oracle):
    """Fang-Oosterlee parameters, T=1: calls decrease in K from 60 to 140 and stay above intrinsic / zero."""
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
