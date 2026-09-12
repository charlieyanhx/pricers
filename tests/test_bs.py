import math

import numpy as np
import pytest

from pricers import bs

CANON = dict(S=100.0, K=100.0, T=1.0, sigma=0.2, r=0.05)


def test_canonical_call_and_put_match_oracle_to_1e8(oracle):
    o = oracle["bs_european"]["canonical"]
    c = bs.price(o["S"], o["K"], o["T"], o["sigma"], "C", o["r"], o["q"])
    p = bs.price(o["S"], o["K"], o["T"], o["sigma"], "P", o["r"], o["q"])
    assert c == pytest.approx(o["call"], abs=o["tol"])
    assert p == pytest.approx(o["put"], abs=o["tol"])
    assert c - p == pytest.approx(o["parity_c_minus_p"], abs=o["tol"])


def test_hull_worked_example_matches_book_and_full_precision(oracle):
    """Hull OFOD, BSM chapter worked example: S=42, K=40, r=10%, sigma=20%, T=0.5 -> 4.76 / 0.81."""
    o = oracle["bs_european"]["hull_example"]
    c = bs.price(o["S"], o["K"], o["T"], o["sigma"], "C", o["r"])
    p = bs.price(o["S"], o["K"], o["T"], o["sigma"], "P", o["r"])
    assert c == pytest.approx(o["hull_rounded"]["call"], abs=o["tol_vs_hull"])
    assert p == pytest.approx(o["hull_rounded"]["put"], abs=o["tol_vs_hull"])
    assert c == pytest.approx(o["call"], abs=1e-9) and p == pytest.approx(o["put"], abs=1e-9)


def test_fang_oosterlee_gbm_table2_calls_match_to_1e8(oracle):
    """Fang & Oosterlee 2008 eq.(51), Table 2: S=100, r=10%, sigma=25%, T=0.1, K in {80,100,120}."""
    o = oracle["bs_european"]["fang_oosterlee_gbm"]
    for K, ref in o["calls"].items():
        assert bs.price(o["S"], float(K), o["T"], o["sigma"], "C", o["r"]) == pytest.approx(ref, abs=o["tol"])


def test_cash_or_nothing_value_is_discounted_n_d2(oracle):
    """Fang & Oosterlee 2008 eq.(52), Table 3: K e^{-rT} N(d2) for S=100, K=120, r=5%, sigma=20%, T=0.1."""
    from scipy.stats import norm

    o = oracle["bs_european"]["fang_oosterlee_cash_or_nothing"]
    _, d2 = bs._d1d2(o["S"], o["K"], o["T"], o["sigma"], o["r"], 0.0)
    assert o["K"] * math.exp(-o["r"] * o["T"]) * norm.cdf(d2) == pytest.approx(o["value"], abs=o["tol"])


def test_put_call_parity_holds_to_1e10_with_dividends():
    S, K, T, r, q, s = 480.0, 500.0, 0.25, 0.03, 0.01, 0.22
    c, p = bs.price(S, K, T, s, "C", r, q), bs.price(S, K, T, s, "P", r, q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_greeks_match_central_finite_differences():
    S, K, T, s, r, q = 500.0, 480.0, 0.12, 0.19, 0.02, 0.01
    g = bs.greeks(S, K, T, s, "P", r, q)
    h = 1e-3
    px = lambda **kw: bs.price(kw.get("S", S), K, kw.get("T", T), kw.get("s", s), "P", kw.get("r", r), q)  # noqa: E731
    assert g.delta == pytest.approx((px(S=S + h) - px(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((px(S=S + h) - 2 * px() + px(S=S - h)) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((px(s=s + 1e-5) - px(s=s - 1e-5)) / 2e-5, rel=1e-6)
    assert g.rho == pytest.approx((px(r=r + 1e-5) - px(r=r - 1e-5)) / 2e-5, rel=1e-6)
    assert g.theta == pytest.approx(px(T=T - 1 / bs.YEAR) - px(), rel=2e-2)   # one calendar day forward


def test_call_put_delta_parity_and_shared_gamma_vega():
    c = bs.greeks(**CANON, right="C")
    p = bs.greeks(**CANON, right="P")
    assert c.delta - p.delta == pytest.approx(1.0, abs=1e-12)
    assert c.gamma == pytest.approx(p.gamma, abs=1e-12) and c.vega == pytest.approx(p.vega, abs=1e-10)


def test_implied_vol_roundtrips_to_1e8_and_is_nan_outside_bounds():
    px = bs.price(500, 520, 0.1, 0.31, "C")
    assert bs.implied_vol(px, 500, 520, 0.1, "C") == pytest.approx(0.31, abs=1e-8)
    assert math.isnan(bs.implied_vol(-1.0, 500, 520, 0.1, "C"))
    assert math.isnan(bs.implied_vol(1e9, 500, 520, 0.1, "C"))
    assert math.isnan(bs.implied_vol(1.0, 500, 520, 0.0, "C"))


def test_expiry_returns_intrinsic_and_step_delta():
    g = bs.greeks(510, 500, 0.0, 0.2, "C")
    assert g.price == 10.0 and g.delta == 1.0 and g.gamma == 0.0
    assert bs.price(490, 500, 0.0, 0.2, "C") == 0.0 and bs.price(490, 500, 0.0, 0.2, "P") == 10.0


def test_array_inputs_broadcast_and_equal_the_scalar_path():
    S = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 1.0, 0.0])
    sig = np.array([0.15, 0.2, 0.25])
    out = bs.price(S, 100.0, T, sig, "C", 0.05, 0.01)
    expected = [bs.price(float(s), 100.0, float(t), float(v), "C", 0.05, 0.01) for s, t, v in zip(S, T, sig, strict=True)]
    np.testing.assert_allclose(out, expected, atol=1e-12)
    g = bs.greeks(S[:2], 100.0, T[:2], sig[:2], "P", 0.05, 0.01)
    for i in range(2):
        gs = bs.greeks(float(S[i]), 100.0, float(T[i]), float(sig[i]), "P", 0.05, 0.01)
        for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert getattr(g, name)[i] == pytest.approx(getattr(gs, name), abs=1e-12)
    ivs = bs.implied_vol_vec(bs.price(S[:2], 100.0, T[:2], sig[:2], "P", 0.05, 0.01), S[:2], 100.0, T[:2], "P", 0.05, 0.01)
    np.testing.assert_allclose(ivs, sig[:2], atol=1e-8)


def test_array_greeks_reject_expired_entries():
    with pytest.raises(ValueError):
        bs.greeks(np.array([100.0, 100.0]), 100.0, np.array([1.0, 0.0]), 0.2, "C")
