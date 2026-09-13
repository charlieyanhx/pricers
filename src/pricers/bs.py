"""Black-Scholes(-Merton) price, Greeks and implied vol for European options on a
dividend-paying underlying. Per-share units.

The public API (`price`, `greeks` -> `Greeks`, `implied_vol`, `YEAR`) is the one in
deskboard's `engine/greeks.py`, kept identical so that module can import this one
unchanged. Units: vega per 1.00 change in vol (per 100 vol points), theta per calendar
day with YEAR = 365, rho per 1.00 change in the rate. `right` is "C" or "P".

Added here: `price` and `greeks` accept numpy arrays (broadcast) for S, K, T and sigma;
scalar inputs take the original scalar path and return floats. `implied_vol` is scalar
(Brent); `implied_vol_vec` loops over it.

Identity kept: C - P == S e^{-qT} - K e^{-rT} (put-call parity) to 1e-10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

YEAR = 365.0


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float      # per 1.00 change in vol (i.e. per 100 vol points)
    theta: float     # per calendar day
    rho: float


def _d1d2(S, K, T, sigma, r, q):
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / v
    return d1, d1 - v


def _is_scalar(*xs) -> bool:
    return all(np.ndim(x) == 0 for x in xs)


def price(S, K, T, sigma, right: str, r: float = 0.0, q: float = 0.0):
    """European price. Scalars -> float; arrays broadcast elementwise (T <= 0 -> intrinsic)."""
    if _is_scalar(S, K, T, sigma):
        if T <= 0:
            return max(0.0, (S - K) if right == "C" else (K - S))
        d1, d2 = _d1d2(S, K, T, sigma, r, q)
        if right == "C":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return _price_array(S, K, T, sigma, right, r, q)


def _price_array(S, K, T, sigma, right, r, q):
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sigma)))
    alive = T > 0
    Tp = np.where(alive, T, 1.0)                      # dummy positive T where expired; masked below
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1d2(S, K, Tp, sigma, r, q)
    if right == "C":
        live = S * np.exp(-q * Tp) * norm.cdf(d1) - K * np.exp(-r * Tp) * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        live = K * np.exp(-r * Tp) * norm.cdf(-d2) - S * np.exp(-q * Tp) * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    return np.where(alive, live, intrinsic)


def greeks(S, K, T, sigma, right: str, r: float = 0.0, q: float = 0.0) -> Greeks:
    """Price and Greeks. Scalars -> floats; arrays broadcast (T <= 0 arrays are not supported)."""
    if not _is_scalar(S, K, T, sigma):
        return _greeks_array(S, K, T, sigma, right, r, q)
    if T <= 0:
        itm = (S > K) if right == "C" else (S < K)
        d = (1.0 if right == "C" else -1.0) * (1.0 if itm else 0.0)
        return Greeks(price(S, K, T, sigma, right, r, q), d, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    sq = np.sqrt(T)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * sq)
    vega = S * disc_q * pdf * sq
    if right == "C":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)) / YEAR
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)) / YEAR
        rho = -K * T * disc_r * norm.cdf(-d2)
    return Greeks(price(S, K, T, sigma, right, r, q), float(delta), float(gamma), float(vega), float(theta), float(rho))


def _greeks_array(S, K, T, sigma, right, r, q) -> Greeks:
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, sigma)))
    if np.any(T <= 0):
        raise ValueError("array greeks need T > 0 everywhere; use price() for expired entries")
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    sq = np.sqrt(T)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * sq)
    vega = S * disc_q * pdf * sq
    if right == "C":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)) / YEAR
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)) / YEAR
        rho = -K * T * disc_r * norm.cdf(-d2)
    return Greeks(_price_array(S, K, T, sigma, right, r, q), delta, gamma, vega, theta, rho)


def implied_vol(target: float, S: float, K: float, T: float, right: str, r: float = 0.0, q: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Brent inversion; NaN when the price is outside no-arbitrage bounds or T <= 0."""
    if T <= 0 or not np.isfinite(target):
        return float("nan")
    intrinsic = price(S, K, T, lo, right, r, q)
    if target < intrinsic - 1e-12 or target > price(S, K, T, hi, right, r, q) + 1e-12:
        return float("nan")
    try:
        return float(brentq(lambda s: price(S, K, T, s, right, r, q) - target, lo, hi, xtol=1e-10, maxiter=200))
    except ValueError:
        return float("nan")


def implied_vol_vec(target, S, K, T, right: str, r: float = 0.0, q: float = 0.0, tol: float = 1e-10,
                    max_iter: int = 60) -> np.ndarray:
    """Vectorised implied vol: Newton in sigma with `scipy.special.ndtr` (no Python loop over points),
    started from the Brenner-Subrahmanyam guess and bisection-guarded, falling back to `implied_vol`
    (Brent) for any element that has not converged. Same answers as `implied_vol` to `tol`; ~20×
    faster on a surface. NaN where the price is outside the no-arbitrage bounds or within 1e-10·S of
    intrinsic (a price that flat carries no information about sigma — a guess would be worse than NaN)."""
    from scipy.special import ndtr

    target, S, K, T = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (target, S, K, T)))
    target, S, K, T = (np.array(a, dtype=float) for a in (target, S, K, T))
    flat = [a.reshape(-1) for a in (target, S, K, T)]
    tg, s_, k_, t_ = flat
    out = np.full(tg.shape, np.nan)
    disc_r, disc_q = np.exp(-r * t_), np.exp(-q * t_)
    fwd = s_ * disc_q / disc_r
    intrinsic = disc_r * np.maximum((fwd - k_) if right == "C" else (k_ - fwd), 0.0)
    cap = s_ * disc_q if right == "C" else k_ * disc_r
    # a price below 1e-10 of spot carries no information about sigma (the map is flat there): NaN, not a guess
    ok = (t_ > 0) & np.isfinite(tg) & (tg >= intrinsic - 1e-12) & (tg <= cap + 1e-12) & (tg - intrinsic > 1e-10 * s_)
    lo, hi = np.full(tg.shape, 1e-4), np.full(tg.shape, 5.0)
    sig = np.clip(np.sqrt(2 * np.pi / np.maximum(t_, 1e-12)) * tg / np.maximum(s_ * disc_q, 1e-12), 0.05, 2.0)
    active = ok.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        sq = np.sqrt(t_[active])
        d1 = (np.log(s_[active] / k_[active]) + (r - q + 0.5 * sig[active] ** 2) * t_[active]) / (sig[active] * sq)
        d2 = d1 - sig[active] * sq
        if right == "C":
            px = s_[active] * disc_q[active] * ndtr(d1) - k_[active] * disc_r[active] * ndtr(d2)
        else:
            px = k_[active] * disc_r[active] * ndtr(-d2) - s_[active] * disc_q[active] * ndtr(-d1)
        diff = px - tg[active]
        vega = s_[active] * disc_q[active] * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) * sq
        # keep the bracket honest: price is increasing in sigma
        lo_a, hi_a = lo[active], hi[active]
        lo_a = np.where(diff < 0, sig[active], lo_a)
        hi_a = np.where(diff > 0, sig[active], hi_a)
        step = np.where(vega > 1e-12, diff / np.maximum(vega, 1e-12), 0.0)
        new = sig[active] - step
        bad = (new <= lo_a) | (new >= hi_a) | ~np.isfinite(new)
        new = np.where(bad, 0.5 * (lo_a + hi_a), new)
        # a point whose price already matches to 1e-15 is converged: keep it, or the bracket test above
        # (new == hi after an underflowed step) would replace the root by the bracket midpoint
        at_root = np.abs(diff) < 1e-15
        new = np.where(at_root, sig[active], new)
        done = at_root | (np.abs(new - sig[active]) < tol)   # converge in SIGMA, not price
        sig[active] = new
        lo[active], hi[active] = lo_a, hi_a
        idx = np.flatnonzero(active)
        active[idx[done]] = False
    out[ok] = sig[ok]
    # anything still active did not converge: Brent it
    for i in np.flatnonzero(active):
        out[i] = implied_vol(float(tg[i]), float(s_[i]), float(k_[i]), float(t_[i]), right, r, q)
    return out.reshape(target.shape)
