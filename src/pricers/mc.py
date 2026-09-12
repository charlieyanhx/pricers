"""Monte Carlo under geometric Brownian motion with a seeded numpy Generator: European
pricing with plain, antithetic, S_T control-variate and combined estimators, and a
Longstaff-Schwartz (2001) American put.

Conventions: per-share prices, continuous r and q, T in years, `right` "C"/"P". Every
estimator returns `MCResult(price, se, ci_lo, ci_hi, n, variant)` where `se` is the standard
error of the estimate and the CI is +-1.96 se. `n` is the number of simulated paths (an
antithetic pair counts as two paths, so variants are comparable per random draw).

Variance reduction:
- antithetic: pairs (Z, -Z); the estimator averages pair means, se from the pair means.
- control variate on S_T: E[S_T] = S e^{(r-q)T}; beta_hat = cov(Y, S_T) / var(S_T) from the
  sample (the optimal coefficient); Y_cv = Y - beta_hat (S_T - E S_T). The residual variance
  ratio 1 - corr^2 is reported as `variance_factor` (oracle: 0.1453 for the reference call).
- both: control variate applied to the antithetic pair means.

LSMC (`american_put_lsm`): 50 exercise dates per year, antithetic paths, regression of the
discounted continuation value on a constant and the first three weighted Laguerre
polynomials of S/K, in-the-money paths only, as in Longstaff & Schwartz (2001); the price is
the mean of the resulting exercise cash flows, so it carries a small low bias. Its `se` is the
paper's convention: the s.e. of that mean *conditional on the exercise rule fitted on the same
paths*, which leaves out the regression's own sampling variance. Measured over 150 seeds at
n = 20,000: on the Table 1 case S = 36 the empirical sd of the estimate is 0.95x the reported
se (95% CI covers the reference 92% of the time); with a large exercise region (S = K = 100,
sigma = 30%, r = 5%, q = 8%) it is 1.16x and the CI covers 86%. Read the LSM se as a floor.

Invariants kept and tested: the sd of the discounted payoff matches its closed form (14.7194
for the reference call, 8.6576 for the put); |estimate - Black-Scholes| < 3 se at fixed seeds;
antithetic and control-variate se are below the plain se on the same draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MCResult", "terminal", "paths", "european", "american_put_lsm", "payoff_sd_exact"]

Z95 = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    price: float
    se: float
    ci_lo: float
    ci_hi: float
    n: int
    variant: str
    variance_factor: float | None = None   # var(Y_cv) / var(Y) when a control variate is used
    beta: float | None = None              # fitted control-variate coefficient


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def terminal(S: float, T: float, sigma: float, r: float, q: float, n: int, seed=0, antithetic: bool = False):
    """S_T for n paths (n even when antithetic: the second half is the mirror of the first)."""
    rng = _rng(seed)
    if antithetic:
        if n % 2:
            raise ValueError("antithetic needs an even n")
        z = rng.standard_normal(n // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n)
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def paths(S: float, T: float, sigma: float, r: float, q: float, n: int, n_steps: int, seed=0,
          antithetic: bool = False) -> np.ndarray:
    """Exact GBM paths on a uniform time grid; shape (n, n_steps + 1), column 0 is S."""
    rng = _rng(seed)
    if antithetic:
        if n % 2:
            raise ValueError("antithetic needs an even n")
        z = rng.standard_normal((n // 2, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n, n_steps))
    dt = T / n_steps
    incr = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    out = np.empty((n, n_steps + 1))
    out[:, 0] = S
    out[:, 1:] = S * np.exp(np.cumsum(incr, axis=1))
    return out


def _payoff(ST: np.ndarray, K: float, right: str) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if right == "C" else np.maximum(K - ST, 0.0)


def european(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0,
             n: int = 100_000, seed=0, antithetic: bool = False, control_variate: bool = False) -> MCResult:
    """Terminal-value Monte Carlo for a European option with optional variance reduction."""
    if n < 4 or (antithetic and n % 2):
        raise ValueError("n >= 4, and even when antithetic")
    ST = terminal(S, T, sigma, r, q, n, seed, antithetic)
    Y = np.exp(-r * T) * _payoff(ST, K, right)
    X = ST
    if antithetic:                        # reduce to pair means: n/2 iid observations
        half = n // 2
        Y = 0.5 * (Y[:half] + Y[half:])
        X = 0.5 * (X[:half] + X[half:])
    variant = {(False, False): "plain", (True, False): "antithetic",
               (False, True): "control_variate", (True, True): "antithetic+control_variate"}[(antithetic, control_variate)]
    vf = beta = None
    if control_variate:
        EX = S * np.exp((r - q) * T)
        cov = np.cov(Y, X, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1])
        Y_cv = Y - beta * (X - EX)
        vf = float(np.var(Y_cv, ddof=1) / np.var(Y, ddof=1))
        Y = Y_cv
    m = len(Y)
    est = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(m))
    return MCResult(est, se, est - Z95 * se, est + Z95 * se, n, variant, vf, beta)


def payoff_sd_exact(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0) -> float:
    """Closed-form standard deviation of the discounted payoff e^{-rT} (S_T - K)^+ (or the put),
    from E[(S_T-K)^+]^2 in terms of lognormal partial moments. Used to check the MC s.e."""
    from scipy.stats import norm

    m = np.log(S) + (r - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    lk = np.log(K)
    # partial moments of the lognormal: E[S_T^k 1{S_T > K}] = exp(k m + k^2 s^2 / 2) N((m + k s^2 - lk)/s)
    def pm(k, above=True):
        z = (m + k * s**2 - lk) / s
        return np.exp(k * m + 0.5 * k**2 * s**2) * (norm.cdf(z) if above else norm.cdf(-z))

    if right == "C":
        e1 = pm(1) - K * pm(0)
        e2 = pm(2) - 2 * K * pm(1) + K**2 * pm(0)
    else:
        e1 = K * pm(0, False) - pm(1, False)
        e2 = K**2 * pm(0, False) - 2 * K * pm(1, False) + pm(2, False)
    return float(np.exp(-r * T) * np.sqrt(e2 - e1**2))


def _laguerre_basis(x: np.ndarray) -> np.ndarray:
    """Constant plus the first three weighted Laguerre polynomials, Longstaff-Schwartz (2001) eq. (4)."""
    w = np.exp(-x / 2)
    L0 = w
    L1 = w * (1 - x)
    L2 = w * (1 - 2 * x + x**2 / 2)
    return np.column_stack([np.ones_like(x), L0, L1, L2])


def american_put_lsm(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0,
                     n: int = 100_000, steps_per_year: int = 50, seed=0, antithetic: bool = True) -> MCResult:
    """Longstaff-Schwartz American put; the reported se is that of the mean exercise cash flow
    over paths (pair means when antithetic), conditional on the fitted exercise rule: the paper's
    convention, and an understatement of up to ~16% when the exercise region is large (module docstring)."""
    n_steps = int(round(steps_per_year * T))
    if n_steps < 1 or n < 100 or (antithetic and n % 2):
        raise ValueError("need steps_per_year * T >= 1, n >= 100 (even when antithetic)")
    P = paths(S, T, sigma, r, q, n, n_steps, seed, antithetic)
    dt = T / n_steps
    disc = np.exp(-r * dt)
    cash = _payoff(P[:, -1], K, "P")               # cash flow at the current optimal stopping time
    for t in range(n_steps - 1, 0, -1):
        cash = cash * disc                          # discounted one step back
        St = P[:, t]
        itm = St < K
        if itm.sum() < 8:
            continue
        A = _laguerre_basis(St[itm] / K)
        coef, *_ = np.linalg.lstsq(A, cash[itm], rcond=None)
        cont = A @ coef
        exercise = _payoff(St[itm], K, "P")
        take = exercise > cont
        idx = np.flatnonzero(itm)[take]
        cash[idx] = exercise[take]
    cash = cash * disc                              # back to t = 0
    Y = cash
    if antithetic:
        half = n // 2
        Y = 0.5 * (cash[:half] + cash[half:])
    est = float(np.mean(Y))
    est = max(est, float(max(K - S, 0.0)))          # immediate exercise at t = 0 dominates if larger
    se = float(np.std(Y, ddof=1) / np.sqrt(len(Y)))
    return MCResult(est, se, est - Z95 * se, est + Z95 * se, n, "lsm")
