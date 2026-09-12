"""Finite-difference pricers on a uniform log-spot grid: explicit, implicit and Crank-Nicolson
theta-schemes with Rannacher start-up, European and American exercise.

PDE in tau = T - t and x = ln S:  V_tau = 1/2 sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V,
central differences in x, theta-scheme in tau (theta = 0 explicit, 1/2 CN, 1 implicit).
Dirichlet boundaries: the European asymptotic values (max'ed with intrinsic for American).

Grid (`Grid`): n_x intervals (n_x rounded up to even), the strike on the centre node, half-width
`width` sigma sqrt(T) plus |ln(S/K)|; the step h is nudged so that S also lands on a node when
|ln(S/K)| >= h/2; the price is read by a cubic spline (exact at nodes). The strike-on-node
placement is what makes CN converge at second order after Rannacher start-up (two implicit
half-steps replace the first CN step). `Grid.n_t` defaults to n_x, except explicit, which
takes the smallest stable count when n_t is not given (dt <= h^2 / sigma^2).

American: `american="psor"` is textbook projected SOR, lexicographic, pure Python (slow: the
reference); `american="bs"` is the Brennan-Schwartz projected-Thomas sweep (elimination from the
top, projected back-substitution from the bottom), exact for puts whose exercise region is
a single interval [0, S*] - the case here - and the fast path used for the 20-case table.
Calls with q > 0 have the region at the top and must use PSOR.

Greeks (`greeks`) are by bump-and-reprice on one fixed grid: S by +-1% (read off the spline),
sigma +-1e-3, r +-1e-4, T - 1/365 for theta per calendar day (same units as `bs.Greeks`).

Invariants kept and tested: European CN error ratio on grid doubling in [3.3, 4.7] (second
order); American >= European; American put >= intrinsic; Brennan-Schwartz == PSOR to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from . import bs
from .bs import Greeks

__all__ = ["FDResult", "Grid", "make_grid", "price", "greeks"]

SCHEMES = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}


@dataclass(frozen=True)
class Grid:
    x: np.ndarray       # log-spot nodes, uniform step h, strike on the centre node
    h: float
    n_t: int
    i_strike: int


@dataclass(frozen=True)
class FDResult:
    price: float
    n_x: int
    n_t: int
    scheme: str
    exercise: str
    american_solver: str | None = None
    psor_iterations: int = 0     # total inner iterations (PSOR only)


def make_grid(S: float, K: float, T: float, sigma: float, n_x: int = 200, n_t: int | None = None,
              width: float = 5.0, scheme: str = "cn") -> Grid:
    n_x = n_x + (n_x % 2)
    half = width * sigma * np.sqrt(T) + abs(np.log(S / K))
    m = n_x // 2
    h = half / m
    k = np.log(S / K) / h
    if abs(k) >= 0.5:                              # nudge h so S sits on a node too
        h = np.log(S / K) / round(k)
    x = np.log(K) + (np.arange(n_x + 1) - m) * h
    if n_t is None:
        n_t = n_x if scheme != "explicit" else int(np.ceil(T * sigma**2 / h**2)) + 1
    return Grid(x, float(h), int(n_t), m)


def _boundary(S_lo: float, S_hi: float, K: float, tau: np.ndarray, right: str, r: float, q: float,
              american: bool) -> tuple[np.ndarray, np.ndarray]:
    """Dirichlet values at both ends for every tau on the time grid."""
    disc_r, disc_q = np.exp(-r * tau), np.exp(-q * tau)
    if right == "C":
        lo, hi = np.zeros_like(tau), S_hi * disc_q - K * disc_r
        if american:
            hi = np.maximum(hi, S_hi - K)
    else:
        lo, hi = K * disc_r - S_lo * disc_q, np.zeros_like(tau)
        if american:
            lo = np.maximum(lo, K - S_lo)
    return lo, hi


def _operator(h: float, sigma: float, r: float, q: float) -> tuple[float, float, float]:
    """Lower / diagonal / upper coefficients of the central-difference operator L."""
    mu = r - q - 0.5 * sigma**2
    a = 0.5 * sigma**2 / h**2 - mu / (2 * h)
    b = -(sigma**2) / h**2 - r
    c = 0.5 * sigma**2 / h**2 + mu / (2 * h)
    return a, b, c


def _step_explicit(V, a, b, c, dt, lo, hi):
    out = np.empty_like(V)
    out[1:-1] = V[1:-1] + dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])
    out[0], out[-1] = lo, hi
    return out


def _rhs(V, a, b, c, dt, theta, lo_new, hi_new):
    """(I + (1-theta) dt L) V^m plus the implicit boundary contributions, interior nodes only."""
    rhs = V[1:-1] + (1 - theta) * dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])
    rhs[0] += theta * dt * a * lo_new
    rhs[-1] += theta * dt * c * hi_new
    return rhs


def _banded(n_int, a, b, c, dt, theta):
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -theta * dt * c
    ab[1, :] = 1.0 - theta * dt * b
    ab[2, :-1] = -theta * dt * a
    return ab


def _psor(ab, rhs, payoff, V0, omega=1.3, tol=1e-9, max_iter=10_000) -> tuple[np.ndarray, int]:
    """Projected SOR, lexicographic order, pure Python loops (the reference, not the fast path)."""
    n = len(rhs)
    upper, diag, lower = ab[0].tolist(), ab[1].tolist(), ab[2].tolist()
    d, g, V = rhs.tolist(), payoff.tolist(), V0.tolist()
    done = 0
    while done < max_iter:
        done += 1
        err = 0.0
        for i in range(n):
            s = d[i]
            if i > 0:
                s -= lower[i - 1] * V[i - 1]
            if i < n - 1:
                s -= upper[i + 1] * V[i + 1]
            v_new = max(g[i], V[i] + omega * (s / diag[i] - V[i]))
            err = max(err, abs(v_new - V[i]))
            V[i] = v_new
        if err < tol:
            break
    return np.asarray(V), done


class _BrennanSchwartz:
    """Projected Thomas sweep for puts: eliminate the super-diagonal from the top once (the
    matrix is time-invariant), then per step solve the bidiagonal system for d' in C and do the
    projected back-substitution from the bottom in a Python loop."""

    def __init__(self, ab: np.ndarray):
        upper, diag, lower = ab[0], ab[1], ab[2]
        n = len(diag)
        bp = np.empty(n)
        m = np.zeros(n)                        # m[i] multiplies d'[i+1]
        bp[-1] = diag[-1]
        for i in range(n - 2, -1, -1):
            m[i] = upper[i + 1] / bp[i + 1]
            bp[i] = diag[i] - m[i] * lower[i]
        self.bp, self.lower_list, self.bp_list = bp, lower.tolist(), bp.tolist()
        self.U = np.zeros((2, n))              # upper-bidiagonal: d'[i] + m[i] d'[i+1] = d[i]
        self.U[0, 1:] = m[:-1]
        self.U[1, :] = 1.0

    def solve(self, rhs: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        dp = solve_banded((0, 1), self.U, rhs).tolist()
        g, lower, bp = payoff.tolist(), self.lower_list, self.bp_list
        V = [0.0] * len(dp)
        prev = 0.0
        for i in range(len(dp)):
            v = (dp[i] - lower[i - 1] * prev) / bp[i] if i > 0 else dp[i] / bp[i]
            prev = V[i] = v if v > g[i] else g[i]
        return np.asarray(V)


def _solve(grid: Grid, S: float, K: float, T: float, sigma: float, right: str, r: float, q: float,
           scheme: str, exercise: str, american: str, rannacher: bool) -> tuple[np.ndarray, int]:
    """Run the time-stepping on `grid`; return the final layer over all nodes and PSOR iterations."""
    theta = SCHEMES[scheme]
    x, h, n_t = grid.x, grid.h, grid.n_t
    Sx = np.exp(x)
    payoff = np.maximum(Sx - K, 0.0) if right == "C" else np.maximum(K - Sx, 0.0)
    is_am = exercise == "american"
    a, b, c = _operator(h, sigma, r, q)
    dt = T / n_t
    # time plan: (dt_step, theta_step); Rannacher replaces the first CN step by two implicit half-steps
    plan = [(dt, theta)] * n_t
    if rannacher and scheme == "cn":
        plan = [(dt / 2, 1.0), (dt / 2, 1.0)] + plan[1:]
    taus = np.cumsum([0.0] + [p[0] for p in plan])
    lo, hi = _boundary(Sx[0], Sx[-1], K, taus, right, r, q, is_am)
    if scheme == "explicit" and dt * sigma**2 / h**2 > 1.0 + 1e-12:
        raise ValueError(f"explicit scheme unstable: need n_t >= {int(np.ceil(T * sigma**2 / h**2))}")
    V = payoff.copy()
    n_int = len(x) - 2
    solvers: dict[float, tuple] = {}
    iters = 0
    for step, (dts, th) in enumerate(plan, start=1):
        if th == 0.0:
            V = _step_explicit(V, a, b, c, dts, lo[step], hi[step])
            if is_am:
                V = np.maximum(V, payoff)
            continue
        key = (dts, th)
        if key not in solvers:
            ab = _banded(n_int, a, b, c, dts, th)
            solvers[key] = (ab, _BrennanSchwartz(ab) if is_am and american == "bs" else None)
        ab, bsolver = solvers[key]
        rhs = _rhs(V, a, b, c, dts, th, lo[step], hi[step])
        if not is_am:
            inner = solve_banded((1, 1), ab, rhs)
        elif american == "bs":
            inner = bsolver.solve(rhs, payoff[1:-1])
        else:
            inner, it = _psor(ab, rhs, payoff[1:-1], V[1:-1])
            iters += it
        V = np.concatenate(([lo[step]], inner, [hi[step]]))
    return V, iters


def _read(grid: Grid, V: np.ndarray, S: float) -> float:
    return float(CubicSpline(grid.x, V)(np.log(S)))


def price(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0,
          n_x: int = 200, n_t: int | None = None, scheme: str = "cn", exercise: str = "european",
          american: str = "bs", rannacher: bool = True, width: float = 5.0, grid: Grid | None = None) -> FDResult:
    """Price by finite differences. `american` in {"bs", "psor"} selects the LCP solver."""
    if scheme not in SCHEMES or right not in ("C", "P") or exercise not in ("european", "american"):
        raise ValueError("scheme in {'explicit','implicit','cn'}, right in {'C','P'}, exercise in {'european','american'}")
    if american not in ("bs", "psor"):
        raise ValueError("american in {'bs', 'psor'}")
    if exercise == "american" and american == "bs" and right == "C":
        raise ValueError("Brennan-Schwartz is implemented for puts; use american='psor' for calls")
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("need T > 0, sigma > 0, S > 0, K > 0")
    g = grid or make_grid(S, K, T, sigma, n_x, n_t, width, scheme)
    V, iters = _solve(g, S, K, T, sigma, right, r, q, scheme, exercise, american, rannacher)
    return FDResult(_read(g, V, S), len(g.x) - 1, g.n_t, scheme, exercise,
                    american if exercise == "american" else None, iters)


def greeks(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0,
           n_x: int = 400, n_t: int | None = None, scheme: str = "cn", exercise: str = "european",
           american: str = "bs", width: float = 5.0) -> Greeks:
    """Bump-and-reprice Greeks on one fixed grid (see module docstring for the bumps)."""
    g = make_grid(S, K, T, sigma, n_x, n_t, width, scheme)

    def px(S_=S, T_=T, sig_=sigma, r_=r) -> float:
        V, _ = _solve(g, S_, K, T_, sig_, right, r_, q, scheme, exercise, american, True)
        return _read(g, V, S_)

    base = px()
    dS, dsig, dr, dT = 0.01 * S, 1e-3, 1e-4, 1.0 / bs.YEAR
    up, dn = px(S_=S + dS), px(S_=S - dS)
    delta = (up - dn) / (2 * dS)
    gamma = (up - 2 * base + dn) / dS**2
    vega = (px(sig_=sigma + dsig) - px(sig_=sigma - dsig)) / (2 * dsig)
    rho = (px(r_=r + dr) - px(r_=r - dr)) / (2 * dr)
    theta = (px(T_=T - dT) - base) if T > dT else 0.0
    return Greeks(base, delta, gamma, vega, theta, rho)
