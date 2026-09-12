"""Lattice pricers: Cox-Ross-Rubinstein and Jarrow-Rudd binomial trees, a Kamrad-Ritchken
trinomial tree, European and American exercise, plus two smoothing/extrapolation helpers.

Units and conventions: per-share prices, continuous rate r and dividend yield q, T in years,
`right` is "C"/"P", `exercise` is "european"/"american". Backward induction is one numpy
expression per step over the whole level (no Python loop over nodes), so N = 5,000 steps
runs in well under a second.

Invariants kept and tested: American >= European for the same tree; American put >= intrinsic;
European CRR converges to Black-Scholes with |error| <= 2.2/N for the S=K=100 reference
call, the sign of the error alternating with the parity of N.

Helpers:
- `bbs=True` (Broadie-Detemple 1996 "binomial Black-Scholes"): the continuation value at the
  penultimate step is the Black-Scholes value over the last dt, which removes the sawtooth.
- `richardson(fn, n)` returns 2 f(2n) - f(n); with `bbs=True` it is BBSR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import bs

__all__ = ["TreeResult", "binomial", "trinomial", "richardson"]


@dataclass(frozen=True)
class TreeResult:
    price: float
    n: int            # number of time steps
    method: str       # "crr" | "crr-ql" | "jr" | "kr-trinomial"
    exercise: str     # "european" | "american"


def _payoff(S: np.ndarray, K: float, right: str) -> np.ndarray:
    return np.maximum(S - K, 0.0) if right == "C" else np.maximum(K - S, 0.0)


def _check(S, K, T, sigma, n, right, exercise):
    if n < 1 or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("need n >= 1, T > 0, sigma > 0, S > 0, K > 0")
    if right not in ("C", "P") or exercise not in ("european", "american"):
        raise ValueError("right in {'C','P'}, exercise in {'european','american'}")


def binomial(S: float, K: float, T: float, sigma: float, right: str, n: int = 500, r: float = 0.0, q: float = 0.0,
             exercise: str = "european", method: str = "crr", bbs: bool = False) -> TreeResult:
    """Recombining binomial tree.

    method: "crr"    Cox-Ross-Rubinstein, u = e^{sigma sqrt dt}, d = 1/u, p = (e^{(r-q)dt} - d)/(u - d) (Hull's form);
            "crr-ql" the same lattice with QuantLib's first-order p = 1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma),
                     included because the measured constants in the oracle fixture were taken from QuantLib;
            "jr"     Jarrow-Rudd, p = 1/2 with the drift in u and d."""
    _check(S, K, T, sigma, n, right, exercise)
    dt = T / n
    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
    elif method == "crr-ql":  # QuantLib's CoxRossRubinstein: same lattice, first-order probability
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = 0.5 + 0.5 * (r - q - 0.5 * sigma**2) * np.sqrt(dt) / sigma
    elif method == "jr":
        drift = (r - q - 0.5 * sigma**2) * dt
        u, d = np.exp(drift + sigma * np.sqrt(dt)), np.exp(drift - sigma * np.sqrt(dt))
        p = 0.5
    else:
        raise ValueError("method in {'crr','crr-ql','jr'}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0,1): increase n")
    disc = np.exp(-r * dt)
    j = np.arange(n + 1)
    # spot at level k, node j (j up-moves): S u^j d^(k-j); level-k nodes are the first k+1 entries
    S_last = S * u**j * d ** (n - j)
    V = _payoff(S_last, K, right)
    start = n - 1
    if bbs:  # Broadie-Detemple: last-step continuation = Black-Scholes over dt, then max with intrinsic
        S_pen = S * u ** j[: n] * d ** (n - 1 - j[: n])
        V = np.asarray(bs.price(S_pen, K, dt, sigma, right, r, q), dtype=float)
        if exercise == "american":
            V = np.maximum(V, _payoff(S_pen, K, right))
        start = n - 2
    for k in range(start, -1, -1):
        V = disc * (p * V[1 : k + 2] + (1.0 - p) * V[: k + 1])
        if exercise == "american":
            V = np.maximum(V, _payoff(S * u ** j[: k + 1] * d ** (k - j[: k + 1]), K, right))
    return TreeResult(float(V[0]), n, method, exercise)


def trinomial(S: float, K: float, T: float, sigma: float, right: str, n: int = 200, r: float = 0.0, q: float = 0.0,
              exercise: str = "european", lam: float = np.sqrt(3.0)) -> TreeResult:
    """Kamrad-Ritchken (1991) trinomial tree: u = e^{lam sigma sqrt dt}, middle branch flat.

    lam = sqrt(3) is the usual choice (p_mid = 2/3); lam = 1 degenerates to CRR."""
    _check(S, K, T, sigma, n, right, exercise)
    dt = T / n
    mu = r - q - 0.5 * sigma**2
    h = lam * sigma * np.sqrt(dt)
    pu = 1.0 / (2 * lam**2) + mu * np.sqrt(dt) / (2 * lam * sigma)
    pd = 1.0 / (2 * lam**2) - mu * np.sqrt(dt) / (2 * lam * sigma)
    pm = 1.0 - pu - pd
    if min(pu, pm, pd) < 0.0:
        raise ValueError("negative branch probability: increase n or lam")
    disc = np.exp(-r * dt)
    # level k has 2k+1 nodes at x = ln S + (i - k) h, i = 0..2k
    i = np.arange(2 * n + 1)
    V = _payoff(S * np.exp((i - n) * h), K, right)
    for k in range(n - 1, -1, -1):
        m = 2 * k + 1
        V = disc * (pu * V[2 : m + 2] + pm * V[1 : m + 1] + pd * V[:m])
        if exercise == "american":
            V = np.maximum(V, _payoff(S * np.exp((i[:m] - k) * h), K, right))
    return TreeResult(float(V[0]), n, "kr-trinomial", exercise)


def richardson(fn: Callable[[int], float], n: int) -> float:
    """Two-point Richardson extrapolation in 1/N: 2 fn(2n) - fn(n). Use with a smoothed tree
    (bbs=True for binomial) so the error is O(1/N) without the parity sawtooth (BBSR)."""
    return 2.0 * fn(2 * n) - fn(n)
