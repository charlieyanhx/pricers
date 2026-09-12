import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def oracle() -> dict:
    return json.loads((FIXTURES / "oracle_values.json").read_text())


@pytest.fixture(scope="session")
def ql():
    """QuantLib module, or skip: the optional [oracle] dependency."""
    return pytest.importorskip("QuantLib")


@pytest.fixture
def ql_bs_process(ql):
    """Factory for a flat Black-Scholes-Merton process; evaluation date 2026-01-01, Actual/365."""
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    def make(S, r, q, sigma):
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)),
        )

    make.today = today
    return make


@pytest.fixture
def ql_option(ql, ql_bs_process):
    """Factory for a vanilla option maturing `T_days` after the evaluation date."""

    def make(right, K, T_days, american=False):
        today = ql_bs_process.today
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if right == "C" else ql.Option.Put, K)
        exercise = ql.AmericanExercise(today, today + T_days) if american else ql.EuropeanExercise(today + T_days)
        return ql.VanillaOption(payoff, exercise)

    return make
