"""`pricers bench`: price the reference European call (S = K = 100, r = 5%, sigma = 20%, T = 1) by
every method at a few resolutions, record |error| against the closed form and wall time, and,
where QuantLib is installed, the same for QuantLib's matching engine at the same resolution.
The table is written between `<!-- bench:start -->` / `<!-- bench:end -->` in README.md.

`check_readme` regenerates the table and compares the method, resolution and |error| columns of
our own deterministic methods with the committed ones to the printed precision (times vary by
machine, QuantLib columns by its version, MC rows by numpy's random stream, so those are excluded);
CI runs it as the zero-spend reproducibility step.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from . import bs, fd, heston, mc, trees

START, END = "<!-- bench:start -->", "<!-- bench:end -->"
CASE = dict(S=100.0, K=100.0, T=1.0, sigma=0.20, r=0.05, q=0.0)
REPEATS = 3


@dataclass(frozen=True)
class Row:
    method: str
    resolution: str
    price: float
    time_ms: float
    ql_price: float | None = None
    ql_time_ms: float | None = None

    @property
    def err(self) -> float:
        return abs(self.price - REF)

    @property
    def ql_err(self) -> float | None:
        return None if self.ql_price is None else abs(self.ql_price - REF)


REF = bs.price(CASE["S"], CASE["K"], CASE["T"], CASE["sigma"], "C", CASE["r"], CASE["q"])


def _timed(fn: Callable[[], float]) -> tuple[float, float]:
    best = float("inf")
    val = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        val = fn()
        best = min(best, time.perf_counter() - t)
    return float(val), best * 1e3


def _ours() -> list[tuple[str, str, Callable[[], float]]]:
    S, K, T, s, r, q = (CASE[k] for k in ("S", "K", "T", "sigma", "r", "q"))
    b = lambda n, m: trees.binomial(S, K, T, s, "C", n, r, q, method=m).price  # noqa: E731
    items: list[tuple[str, str, Callable[[], float]]] = [
        ("Black-Scholes closed form", "-", lambda: bs.price(S, K, T, s, "C", r, q)),
    ]
    items += [("CRR binomial", f"N={n}", (lambda n=n: b(n, "crr"))) for n in (100, 1000, 5000)]
    items += [("Jarrow-Rudd binomial", f"N={n}", (lambda n=n: b(n, "jr"))) for n in (100, 1000)]
    items += [("Kamrad-Ritchken trinomial", f"N={n}", (lambda n=n: trees.trinomial(S, K, T, s, "C", n, r, q).price))
              for n in (100, 1000)]
    items += [("BBS + Richardson (BBSR)", "N=100,200",
               lambda: trees.richardson(lambda n: trees.binomial(S, K, T, s, "C", n, r, q, bbs=True).price, 100))]
    items += [("FD explicit (n_t at stability)", f"n_x={n}", (lambda n=n: fd.price(S, K, T, s, "C", r, q, n_x=n, scheme="explicit").price))
              for n in (100, 400)]
    items += [("FD implicit", f"n_x=n_t={n}", (lambda n=n: fd.price(S, K, T, s, "C", r, q, n_x=n, scheme="implicit").price))
              for n in (100, 400)]
    items += [("FD Crank-Nicolson + Rannacher", f"n_x=n_t={n}", (lambda n=n: fd.price(S, K, T, s, "C", r, q, n_x=n, scheme="cn").price))
              for n in (100, 400, 1600)]
    items += [("MC plain (seed 42)", f"n={n:,}", (lambda n=n: mc.european(S, K, T, s, "C", r, q, n=n, seed=42).price))
              for n in (100_000, 1_000_000)]
    items += [("MC antithetic + S_T control variate (seed 42)", "n=100,000",
               lambda: mc.european(S, K, T, s, "C", r, q, n=100_000, seed=42, antithetic=True, control_variate=True).price)]
    items += [("COS (GBM characteristic function)", f"N={n}", (lambda n=n: heston.price_gbm(S, K, T, s, "C", r, q, N=n)))
              for n in (64, 256)]
    return items


def _quantlib() -> dict[tuple[str, str], Callable[[], float]] | None:
    """Matching QuantLib engines keyed like `_ours()` rows; None when QuantLib is absent."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    S, K, T, s, r, q = (CASE[k] for k in ("S", "K", "T", "sigma", "r", "q"))
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), s, dc)),
    )
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(today + 365))

    def npv(engine) -> Callable[[], float]:
        def run() -> float:
            opt.setPricingEngine(engine)
            return opt.NPV()
        return run

    out: dict[tuple[str, str], Callable[[], float]] = {
        ("Black-Scholes closed form", "-"): npv(ql.AnalyticEuropeanEngine(proc)),
    }
    for n in (100, 1000, 5000):
        out[("CRR binomial", f"N={n}")] = npv(ql.BinomialVanillaEngine(proc, "crr", n))
    for n in (100, 1000):
        out[("Jarrow-Rudd binomial", f"N={n}")] = npv(ql.BinomialVanillaEngine(proc, "jarrowrudd", n))
    for n in (100, 400, 1600):
        out[("FD Crank-Nicolson + Rannacher", f"n_x=n_t={n}")] = npv(ql.FdBlackScholesVanillaEngine(proc, n, n))
    for n in (100_000, 1_000_000):
        out[("MC plain (seed 42)", f"n={n:,}")] = npv(
            ql.MCEuropeanEngine(proc, "pseudorandom", timeSteps=1, requiredSamples=n, seed=42))
    return out


def run() -> list[Row]:
    ql_engines = _quantlib()
    rows = []
    for method, res, fn in _ours():
        price, ms = _timed(fn)
        ql_price = ql_ms = None
        if ql_engines and (method, res) in ql_engines:
            ql_price, ql_ms = _timed(ql_engines[(method, res)])
        rows.append(Row(method, res, price, ms, ql_price, ql_ms))
    return rows


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def _fmt_err(e: float | None) -> str:
    return "-" if e is None else f"{e:.1e}"


def _machine() -> str:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    os_name = f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else f"{platform.system()} {platform.release()}"
    try:
        import QuantLib as ql
        qlv = f"QuantLib {ql.__version__}"
    except ImportError:
        qlv = "QuantLib not installed"
    return (f"{cpu}, {os_name}, Python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, {qlv}")


def table(rows: list[Row]) -> str:
    lines = [
        f"Reference call S=K=100, r=5%, sigma=20%, T=1: closed form {REF:.10f}. Generated by `pricers bench` on "
        f"{_machine()}; wall time is the best of {REPEATS} runs; QuantLib columns are its engine of the same kind at "
        "the same resolution (`-` where none matches).",
        "",
        "| method | resolution | price | abs error | time | QuantLib price | QuantLib abs error | QuantLib time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        qp = "-" if r.ql_price is None else f"{r.ql_price:.6f}"
        lines.append(f"| {r.method} | {r.resolution} | {r.price:.6f} | {_fmt_err(r.err)} | {_fmt_ms(r.time_ms)} | "
                     f"{qp} | {_fmt_err(r.ql_err)} | {_fmt_ms(r.ql_time_ms)} |")
    return "\n".join(lines)


def write_readme(readme: Path, text: str) -> None:
    src = readme.read_text()
    if START not in src or END not in src:
        raise SystemExit(f"{readme}: markers {START} / {END} not found")
    head, rest = src.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{text}\n{END}{tail}")


def _error_cells(text: str) -> list[tuple[str, str, float]]:
    """(method, resolution, |error|) per table row, MC rows excluded (their value is the seed's, not the method's)."""
    out = []
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| method"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not cells[0].startswith("MC"):
                out.append((cells[0], cells[1], float(cells[3])))
    return out


def _same(a: float, b: float) -> bool:
    """Equal to the printed precision, allowing last-bit platform differences: 5% relative, or both < 1e-12."""
    return (a < 1e-12 and b < 1e-12) or abs(a - b) <= 0.05 * max(a, b)


def check_readme(readme: Path, rows: list[Row]) -> bool:
    src = readme.read_text()
    committed = _error_cells(src.split(START, 1)[1].split(END, 1)[0])
    fresh = _error_cells(table(rows))
    ok = len(committed) == len(fresh)
    if not ok:
        print(f"drift: {len(committed)} committed rows vs {len(fresh)} regenerated")
    for c, f in zip(committed, fresh, strict=False):
        if c[:2] != f[:2] or not _same(c[2], f[2]):
            print(f"drift: committed {c} != regenerated {f}")
            ok = False
    return ok
