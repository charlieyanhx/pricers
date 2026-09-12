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


def implied_vol_vec(target, S, K, T, right: str, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """Elementwise `implied_vol` over broadcast inputs (a loop; Brent is per element)."""
    target, S, K, T = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (target, S, K, T)))
    out = np.empty(target.shape)
    for idx in np.ndindex(target.shape):
        out[idx] = implied_vol(float(target[idx]), float(S[idx]), float(K[idx]), float(T[idx]), right, r, q)
    return out
