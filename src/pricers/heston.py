"""COS method (Fang & Oosterlee 2008) for European options under Heston and, as a check on the
machinery, under Black-Scholes. The Heston characteristic function is the Albrecher et al. (2007)
/ Gatheral form, which is continuous in the complex log for every maturity (the "little Heston
trap"); the Fang-Oosterlee test parameters violate the Feller condition, so this matters.

Conventions: per-share prices, continuous r and q, T in years, `right` "C"/"P". Calls are
priced as put + S e^{-qT} - K e^{-rT} (parity), because the put's COS coefficients are bounded
on the truncation range while the call's grow with e^b, as the paper recommends.

Truncation range in y = ln(S_T/K): [a, b] = x + c1 -+ L sqrt(c2 + sqrt(c4)) with x = ln(S/K) and
c1, c2, c4 the cumulants of ln(S_T/S_0) (the paper's eq. (49): the cumulants of ln(S_T/K) are those
of ln(S_T/S_0) shifted by x). c1 and c2 are in closed form, c4 by a five-point stencil on log phi at
u = 0 (`c4="numeric"`, the default; `c4=0` reproduces the plain c1 -+ L sqrt(c2) range). The put's
payoff support is [a, min(b, 0)] and the put is 0 when a >= 0, so the range holds for any
moneyness (a range placed on c1 alone, without x, leaks the density off [a, b] once |x| exceeds
about 4.5 sigma sqrt(T) and returns negative calls; a test at 6-10 standard deviations keeps this).

Defaults N = 1024, L = 12 (the GBM check `price_gbm` keeps N = 256, L = 10, the paper's GBM
setting). Measured on the paper's T = 1 case: with c4 = 0 the error saturates at -3.9e-5 for any N
(QuantLib's COSHestonEngine at L = 12, N = 256 gives the same -3.9e-5), with the c4 term it is
4.5e-8 already at N = 256. N = 256 is not enough in general: for Feller-violated, high vol-of-vol
parameters (sigma_v ~ 0.9, kappa < 1, T ~ 1-2 y) the c4 term widens the range about 2.5x and
256 terms leave errors of 1e-3 near the money, which N = 1024 brings below 1e-7 (test:
`test_feller_violated_high_vol_of_vol_needs_1024_terms`). Very heavy tails (the Andersen 2008
case, sigma_v = 1, T = 10) make the numerical c4 unusable and need c4=0 with L ~ 30 and N ~ 2048
instead; see the tests.

Invariants kept and tested: COS under GBM matches Black-Scholes to 1e-8 at the money and to
1e-10 at 6-10 standard deviations from it; Heston call prices match the paper's eq. (53) values
(T = 1: 5.785155450, T = 10: 22.318945791) to 1e-6 and QuantLib's COSHestonEngine when
installed; put-call parity holds by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["HestonParams", "cf_heston", "cf_gbm", "cumulant4_numeric", "cumulants_heston", "cumulants_gbm", "cos_price",
           "price", "price_gbm"]


@dataclass(frozen=True)
class HestonParams:
    v0: float        # initial variance
    kappa: float     # mean-reversion speed
    theta: float     # long-run variance
    sigma_v: float   # vol of variance
    rho: float       # correlation of spot and variance Brownian motions

    def feller(self) -> bool:
        return 2 * self.kappa * self.theta > self.sigma_v**2


def cf_heston(u: np.ndarray, T: float, p: HestonParams, r: float, q: float) -> np.ndarray:
    """Characteristic function of ln(S_T / S_0) in the Albrecher/Gatheral form."""
    iu = 1j * u
    beta = p.kappa - p.rho * p.sigma_v * iu
    D = np.sqrt(beta**2 + p.sigma_v**2 * (u**2 + iu))
    G = (beta - D) / (beta + D)
    eDT = np.exp(-D * T)
    A = iu * (r - q) * T + p.kappa * p.theta / p.sigma_v**2 * ((beta - D) * T - 2 * np.log((1 - G * eDT) / (1 - G)))
    B = (beta - D) / p.sigma_v**2 * (1 - eDT) / (1 - G * eDT)
    return np.exp(A + B * p.v0)


def cf_gbm(u: np.ndarray, T: float, sigma: float, r: float, q: float) -> np.ndarray:
    """Characteristic function of ln(S_T / S_0) under geometric Brownian motion."""
    return np.exp(1j * u * (r - q - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)


def cumulant4_numeric(cf: Callable[[np.ndarray], np.ndarray], h: float = 0.05) -> float:
    """Fourth cumulant from a five-point stencil on log cf at u = 0 (O(h^2); enough for a range)."""
    f = np.log(cf(np.array([-2 * h, -h, 0.0, h, 2 * h], dtype=complex)))
    return float(np.real((f[4] - 4 * f[3] + 6 * f[2] - 4 * f[1] + f[0]) / h**4))


def cumulants_heston(T: float, p: HestonParams, r: float, q: float, c4="numeric") -> tuple[float, float, float]:
    """c1, c2 (mean, variance) of ln(S_T/S_0) in closed form; c4 numeric (default) or a given number.

    c1 is Fang & Oosterlee (2008) Table 11. c2 is Var(X_T) = E[I] + Var(I)/4 - (rho/eta)(Cov(I, v_T)
    + lambda Var(I)) with I = int_0^T v_t dt, integrated in closed form from the CIR covariance
    Cov(v_s, v_t) = e^{-lambda(t-s)} Var(v_s). It agrees with the numerical second derivative of
    log phi at u = 0 to 1e-9; the paper's printed c2, as transcribed here, is smaller by
    eta^2 theta (1 - e^{-lambda T}) / (4 lambda^3) (about 2% for the paper's test parameters)."""
    lam, eta, ub, u0, rho, mu = p.kappa, p.sigma_v, p.theta, p.v0, p.rho, r - q
    e1, e2 = np.exp(-lam * T), np.exp(-2 * lam * T)
    c1 = mu * T + (1 - e1) * (ub - u0) / (2 * lam) - 0.5 * ub * T
    c2 = (1 / (8 * lam**3)) * (
        eta**2 * (ub - 2 * u0) * e2
        + 4 * e1 * (T * eta**2 * lam * (ub - u0) + 2 * T * eta * lam**2 * rho * (u0 - ub) + eta**2 * ub
                    + 2 * eta * lam * rho * u0 - 4 * eta * lam * rho * ub + 2 * lam**2 * (ub - u0))
        + 2 * T * eta**2 * lam * ub - 8 * T * eta * lam**2 * rho * ub + 8 * T * lam**3 * ub
        + 2 * eta**2 * u0 - 5 * eta**2 * ub - 8 * eta * lam * rho * u0 + 16 * eta * lam * rho * ub
        + 8 * lam**2 * (u0 - ub)
    )
    if c4 == "numeric":
        c4 = max(cumulant4_numeric(lambda u: cf_heston(u, T, p, r, q)), 0.0)
    return float(c1), float(c2), float(c4)


def cumulants_gbm(T: float, sigma: float, r: float, q: float) -> tuple[float, float, float]:
    return (r - q - 0.5 * sigma**2) * T, sigma**2 * T, 0.0


def _chi_psi(k: np.ndarray, a: float, b: float, c: float, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-series integrals of e^y and 1 over [c, d], Fang & Oosterlee eqs. (22)-(23)."""
    w = k * np.pi / (b - a)
    chi = (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
           + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w**2)
    psi = np.empty_like(w)
    psi[0] = d - c
    psi[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
    return chi, psi


def cos_price(cf: Callable[[np.ndarray], np.ndarray], S: float, K: float, T: float, right: str, r: float, q: float,
              cumulants: tuple[float, float, float], N: int = 1024, L: float = 12.0) -> float:
    """Generic COS price for a vanilla; `cf(u)` is the characteristic function of ln(S_T/S_0)."""
    c1, c2, c4 = cumulants
    x = np.log(S / K)
    w = L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))
    a, b = x + c1 - w, x + c1 + w                 # range in y = ln(S_T/K): cumulants of ln(S_T/S_0) shifted by x
    if a >= 0.0:                                  # the whole density sits above the strike: the put is worthless
        put = 0.0
    else:
        k = np.arange(N)
        chi, psi = _chi_psi(k, a, b, a, min(b, 0.0))  # put: payoff support [a, min(b, 0)] in y
        U = 2.0 / (b - a) * (-chi + psi)          # V_k / K for the put
        u = k * np.pi / (b - a)
        terms = np.real(cf(u) * np.exp(1j * u * (x - a))) * U
        terms[0] *= 0.5
        put = K * np.exp(-r * T) * np.sum(terms)
    if right == "P":
        return float(put)
    return float(put + S * np.exp(-q * T) - K * np.exp(-r * T))


def price(S: float, K: float, T: float, right: str, params: HestonParams, r: float = 0.0, q: float = 0.0,
          N: int = 1024, L: float = 12.0, c4="numeric") -> float:
    """European Heston price by COS with N terms and range parameter L (see module docstring)."""
    if T <= 0 or S <= 0 or K <= 0 or right not in ("C", "P"):
        raise ValueError("need T > 0, S > 0, K > 0, right in {'C','P'}")
    return cos_price(lambda u: cf_heston(u, T, params, r, q), S, K, T, right, r, q,
                     cumulants_heston(T, params, r, q, c4), N, L)


def price_gbm(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0,
              N: int = 256, L: float = 10.0) -> float:
    """Black-Scholes price by COS (the machinery check; L = 10 as in the paper's GBM tests)."""
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0 or right not in ("C", "P"):
        raise ValueError("need T > 0, S > 0, K > 0, sigma > 0, right in {'C','P'}")
    return cos_price(lambda u: cf_gbm(u, T, sigma, r, q), S, K, T, right, r, q, cumulants_gbm(T, sigma, r, q), N, L)
