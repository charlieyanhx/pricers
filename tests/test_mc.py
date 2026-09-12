import numpy as np
import pytest

from pricers import bs, mc

CASE = dict(S=100.0, K=100.0, T=1.0, sigma=0.2, r=0.05)
REF_CALL = bs.price(100.0, 100.0, 1.0, 0.2, "C", 0.05)
REF_PUT = bs.price(100.0, 100.0, 1.0, 0.2, "P", 0.05)


def _eu(right="C", **kw):
    return mc.european(100.0, 100.0, 1.0, 0.2, right, 0.05, **kw)


def test_closed_form_payoff_sd_matches_oracle_to_1e6(oracle):
    """Exact sd of the discounted payoff on the canonical case (fixture: 14.719404 call, 8.65758 put; lognormal
    partial moments recomputed at value-collection time)."""
    o = oracle["convergence_constants_S100_K100_r5_sig20_T1_call"]["mc_exact_sd_discounted_payoff"]
    assert mc.payoff_sd_exact(100.0, 100.0, 1.0, 0.2, "C", 0.05) == pytest.approx(o["call"], abs=1e-6)
    assert mc.payoff_sd_exact(100.0, 100.0, 1.0, 0.2, "P", 0.05) == pytest.approx(o["put"], abs=1e-6)


def test_plain_se_times_sqrt_n_is_within_10pct_of_14_7194_at_n_200k(oracle):
    """Plain MC s.e. x sqrt(n) vs the exact sd of the discounted payoff (fixture), call and put, seed 7."""
    o = oracle["convergence_constants_S100_K100_r5_sig20_T1_call"]["mc_exact_sd_discounted_payoff"]
    for right, sd in (("C", o["call"]), ("P", o["put"])):
        res = _eu(right, n=200_000, seed=7)
        assert res.se * np.sqrt(res.n) == pytest.approx(sd, rel=0.10)
        assert res.variant == "plain" and res.n == 200_000


def test_control_variate_variance_factor_is_within_0_12_to_0_18(oracle):
    """S_T control variate on the canonical call: residual variance ratio vs the fixture's 1 - corr^2 = 0.1453
    (corr 0.9245) and its s.e. x sqrt(n) vs the fixture's 5.6106."""
    o = oracle["convergence_constants_S100_K100_r5_sig20_T1_call"]["mc_exact_sd_discounted_payoff"]["control_variate_S_T"]
    res = _eu(n=200_000, seed=7, control_variate=True)
    assert 0.12 <= res.variance_factor <= 0.18
    assert res.variance_factor == pytest.approx(o["variance_factor"], abs=0.02)
    assert res.se * np.sqrt(res.n) == pytest.approx(o["sd_call"], rel=0.10)


def test_antithetic_equivalent_per_draw_sd_matches_oracle_within_10pct(oracle):
    """Antithetic s.e. x sqrt(n) on the canonical call vs the fixture's 10.4 (measured at value-collection time)."""
    o = oracle["convergence_constants_S100_K100_r5_sig20_T1_call"]["mc_exact_sd_discounted_payoff"]
    res = _eu(n=200_000, seed=7, antithetic=True)
    assert res.se * np.sqrt(res.n) == pytest.approx(o["antithetic_equiv_per_draw_sd_call"], rel=0.10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_variant_lands_within_3_se_of_black_scholes(seed):
    for av in (False, True):
        for cv in (False, True):
            for right, ref in (("C", REF_CALL), ("P", REF_PUT)):
                res = _eu(right, n=200_000, seed=seed, antithetic=av, control_variate=cv)
                assert abs(res.price - ref) < 3 * res.se, (res.variant, right, seed)
                assert res.ci_lo < res.price < res.ci_hi
                assert res.ci_hi - res.ci_lo == pytest.approx(2 * 1.959963984540054 * res.se, abs=1e-12)


def test_variance_reduction_orders_se_plain_gt_antithetic_gt_cv_gt_both():
    plain = _eu(n=200_000, seed=11).se
    av = _eu(n=200_000, seed=11, antithetic=True).se
    cv = _eu(n=200_000, seed=11, control_variate=True).se
    both = _eu(n=200_000, seed=11, antithetic=True, control_variate=True).se
    assert plain > av > cv > both


def test_paths_are_exact_gbm_with_the_right_terminal_moments():
    P = mc.paths(100.0, 1.0, 0.2, 0.05, 0.0, 200_000, 4, seed=3, antithetic=True)
    assert P.shape == (200_000, 5) and np.all(P[:, 0] == 100.0)
    ST = P[:, -1]
    assert np.mean(ST) == pytest.approx(100.0 * np.exp(0.05), rel=2e-3)
    assert np.std(np.log(ST / 100.0)) == pytest.approx(0.2, rel=1e-2)
    same = mc.paths(100.0, 1.0, 0.2, 0.05, 0.0, 200_000, 4, seed=3, antithetic=True)
    assert np.array_equal(P, same)


def test_lsm_reproduces_longstaff_schwartz_table1_within_sampling_error(oracle):
    """All 20 Table 1 cases, 100,000 antithetic paths, 50 exercise dates/yr, seed 1.

    Bars: |ours - paper LSM| <= 3 sqrt(se_ours^2 + se_paper^2) per case (both are noisy estimates
    of the same low-biased quantity); |ours - continuous American| < 0.03 per case (measured worst
    at this seed 0.0201; the paper's own LSM column is up to 0.027 below the continuous value);
    ours >= European; mean signed deviation from the continuous value in (-0.02, 0)."""
    o = oracle["american_put_lsm2001_table1"]
    devs = []
    for row in o["rows"]:
        res = mc.american_put_lsm(row["S"], o["K"], row["T"], row["sigma"], o["r"], n=100_000, seed=1)
        se_comb = np.hypot(res.se, row["lsm_se"])
        assert abs(res.price - row["lsm_sim"]) <= 3 * se_comb, row
        assert abs(res.price - row["american_continuous"]) < 0.03, row
        assert res.price >= row["european_bs"] - 3 * res.se
        assert res.variant == "lsm" and res.se < 0.02
        devs.append(res.price - row["american_continuous"])
    assert -0.02 < np.mean(devs) < 0.0


def test_lsm_meets_the_literal_bars_on_the_first_table_case(oracle):
    """S=36, sigma=0.2, T=1: within 2 paper-s.e. of 4.472 and within 0.02 of 4.4867, deterministic seed."""
    o = oracle["american_put_lsm2001_table1"]
    row = o["rows"][0]
    res = mc.american_put_lsm(row["S"], o["K"], row["T"], row["sigma"], o["r"], n=100_000, seed=1)
    assert abs(res.price - row["lsm_sim"]) <= 2 * row["lsm_se"]
    assert abs(res.price - row["american_continuous"]) < 0.02


def test_lsm_never_prices_below_intrinsic_deep_in_the_money():
    res = mc.american_put_lsm(20.0, 40.0, 1.0, 0.2, 0.06, n=20_000, seed=5)
    assert res.price >= 20.0


def test_rejects_odd_n_with_antithetic_and_tiny_n():
    with pytest.raises(ValueError):
        _eu(n=1001, antithetic=True)
    with pytest.raises(ValueError):
        _eu(n=2)
    with pytest.raises(ValueError):
        mc.american_put_lsm(36.0, 40.0, 0.001, 0.2, 0.06, n=1000)
