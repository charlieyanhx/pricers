import pytest

from pricers import bs, trees

CANON = dict(S=100.0, K=100.0, T=1.0, sigma=0.2, r=0.05)
REF_CALL = bs.price(100.0, 100.0, 1.0, 0.2, "C", 0.05)


def _crr(n, method="crr", right="C", exercise="european"):
    return trees.binomial(100.0, 100.0, 1.0, 0.2, right, n, 0.05, exercise=exercise, method=method).price


def test_crr_error_bounded_by_2_2_over_n_and_sign_flips_between_n_and_n_plus_1(oracle):
    """Oracle: |err| <= 2.2/N for N >= 10; the sign alternates with the parity of N."""
    bound = 2.2
    for n in (50, 100, 200, 400, 800):
        e_even, e_odd = _crr(n) - REF_CALL, _crr(n + 1) - REF_CALL
        assert abs(e_even) <= bound / n and abs(e_odd) <= bound / (n + 1)
        assert e_even < 0 < e_odd


def test_crr_n_times_error_limits_measured_for_both_probability_forms(oracle):
    """QuantLib's first-order p gives the fixture constants -2.062 / +1.690; the exact-p textbook CRR
    gives -2.000 / +1.753 (measured here, same lattice, different p)."""
    c = oracle["convergence_constants_S100_K100_r5_sig20_T1_call"]["crr_N_times_error"]
    assert (_crr(800, "crr-ql") - REF_CALL) * 800 == pytest.approx(c["even_N"], abs=0.01)
    assert (_crr(801, "crr-ql") - REF_CALL) * 801 == pytest.approx(c["odd_N"], abs=0.01)
    assert (_crr(800) - REF_CALL) * 800 == pytest.approx(-2.000, abs=0.01)
    assert (_crr(801) - REF_CALL) * 801 == pytest.approx(1.753, abs=0.01)


def test_crr_error_halves_when_n_doubles_at_fixed_parity():
    e1, e2 = _crr(400) - REF_CALL, _crr(800) - REF_CALL
    assert e2 / e1 == pytest.approx(0.5, abs=0.02)


def test_hull_five_step_american_put_is_4_49(oracle):
    """Hull OFOD, Basic Numerical Procedures: S=K=50, r=10%, sigma=40%, T=5/12, 5 steps -> 4.49."""
    o = oracle["hull_american_put_tree"]
    p = trees.binomial(o["S"], o["K"], o["T"], o["sigma"], "P", 5, o["r"], exercise="american").price
    assert p == pytest.approx(o["hull_rounded_5_steps"], abs=5e-3)


def test_hull_american_put_converges_to_4_2842_at_large_n(oracle):
    o = oracle["hull_american_put_tree"]
    for method in ("crr", "jr"):
        p = trees.binomial(o["S"], o["K"], o["T"], o["sigma"], "P", 5000, o["r"], exercise="american", method=method).price
        assert p == pytest.approx(o["converged"], abs=1e-3)
    t = trees.trinomial(o["S"], o["K"], o["T"], o["sigma"], "P", 2000, o["r"], exercise="american").price
    assert t == pytest.approx(o["converged"], abs=1e-3)


def test_bbsr_beats_plain_crr_by_two_orders_on_the_hull_put(oracle):
    o = oracle["hull_american_put_tree"]
    plain = trees.binomial(o["S"], o["K"], o["T"], o["sigma"], "P", 100, o["r"], exercise="american").price
    bbsr = trees.richardson(
        lambda n: trees.binomial(o["S"], o["K"], o["T"], o["sigma"], "P", n, o["r"], exercise="american", bbs=True).price, 100)
    assert abs(bbsr - o["converged"]) < 5e-4 < abs(plain - o["converged"]) / 10


def test_european_trees_agree_with_black_scholes_at_stated_tolerances():
    for right in ("C", "P"):
        ref = bs.price(100.0, 100.0, 1.0, 0.2, right, 0.05)
        assert _crr(2000, right=right) == pytest.approx(ref, abs=1.1e-3)          # 2.2 / N
        assert _crr(2000, "jr", right=right) == pytest.approx(ref, abs=2e-3)
        assert trees.trinomial(100.0, 100.0, 1.0, 0.2, right, 2000, 0.05).price == pytest.approx(ref, abs=1.1e-3)


def test_american_dominates_european_and_intrinsic():
    for S in (80.0, 100.0, 120.0):
        for method in ("crr", "jr"):
            am = trees.binomial(S, 100.0, 1.0, 0.2, "P", 400, 0.05, exercise="american", method=method).price
            eu = trees.binomial(S, 100.0, 1.0, 0.2, "P", 400, 0.05, method=method).price
            assert am >= eu - 1e-12 and am >= max(100.0 - S, 0.0) - 1e-12
        am_t = trees.trinomial(S, 100.0, 1.0, 0.2, "P", 400, 0.05, exercise="american").price
        assert am_t >= trees.trinomial(S, 100.0, 1.0, 0.2, "P", 400, 0.05).price - 1e-12


def test_american_call_without_dividends_equals_european():
    am = trees.binomial(100.0, 100.0, 1.0, 0.2, "C", 500, 0.05, exercise="american").price
    assert am == pytest.approx(_crr(500), abs=1e-12)


def test_result_carries_n_method_and_exercise():
    res = trees.binomial(100.0, 100.0, 1.0, 0.2, "C", 7, 0.05, exercise="american", method="jr")
    assert (res.n, res.method, res.exercise) == (7, "jr", "american")


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        trees.binomial(100.0, 100.0, 1.0, 0.2, "C", 0, 0.05)
    with pytest.raises(ValueError):
        trees.binomial(100.0, 100.0, 1.0, 0.2, "X", 10, 0.05)
    with pytest.raises(ValueError):
        trees.binomial(100.0, 100.0, 1.0, 0.2, "C", 10, 0.05, method="lr")


def test_crr_ql_and_jr_match_quantlib_binomial_engines_to_1e10(ql, ql_bs_process, ql_option):
    """Same lattice and same probability as QuantLib's CoxRossRubinstein / JarrowRudd: identical to rounding."""
    proc = ql_bs_process(100.0, 0.05, 0.0, 0.2)
    for n in (50, 51, 200):
        for ours, theirs in (("crr-ql", "crr"), ("jr", "jarrowrudd")):
            opt = ql_option("C", 100.0, 365)
            opt.setPricingEngine(ql.BinomialVanillaEngine(proc, theirs, n))
            assert _crr(n, ours) == pytest.approx(opt.NPV(), abs=1e-10)
