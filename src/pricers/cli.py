"""`pricers bench`, `pricers report` and `pricers price --method ... --S ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

METHODS = ("bs", "crr", "jr", "trinomial", "fd-explicit", "fd-implicit", "fd-cn", "mc", "mc-cv", "lsm", "cos-gbm")
_REPO = Path(__file__).resolve().parents[2]
README = _REPO / "README.md"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pricers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="accuracy-vs-speed table for the reference call; rewrites README.md by default")
    b.add_argument("--readme", default=str(README))
    b.add_argument("--no-write", action="store_true", help="print the table only")
    b.add_argument("--check", action="store_true", help="regenerate and compare the |error| columns with README.md")
    rp = sub.add_parser("report", help="validation and convergence-rate tables; rewrites README.md by default")
    rp.add_argument("--readme", default=str(README))
    rp.add_argument("--no-write", action="store_true", help="print the tables only")
    cal = sub.add_parser("calibrate", help="Heston calibration demo: recover known parameters from a synthetic surface")
    cal.add_argument("--noise", type=float, default=0.0, help="Gaussian noise added to the surface, in vol (0.002 = 20 bp)")
    cal.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("price", help="price one option by one method")
    p.add_argument("--method", choices=METHODS, default="bs")
    p.add_argument("--S", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--right", choices=("C", "P"), default="C")
    p.add_argument("--american", action="store_true", help="American exercise (trees, fd-*, lsm)")
    p.add_argument("--n", type=int, default=None, help="steps / grid intervals / paths / COS terms")
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "bench":
        from . import bench
        rows = bench.run()
        text = bench.table(rows)
        print(text)
        readme = Path(a.readme)
        if a.check:
            ok = bench.check_readme(readme, rows)
            print("bench check: " + ("ok" if ok else "DRIFT"))
            raise SystemExit(0 if ok else 1)
        if not a.no_write:
            bench.write_readme(readme, text)
            print(f"wrote the table to {readme}")
    elif a.cmd == "report":
        from . import report
        text = report.tables()
        print(text)
        if not a.no_write:
            report.write_readme(Path(a.readme), text)
            print(f"wrote the tables to {a.readme}")
    elif a.cmd == "calibrate":
        import time

        import numpy as np

        from .calibrate import Surface, calibrate
        from .heston import HestonParams

        true = HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma_v=0.6, rho=-0.7)
        surf = Surface.from_params(100.0, 0.02, 0.01, [70, 80, 90, 95, 100, 105, 110, 120, 130], [0.1, 0.25, 0.5, 1.0, 2.0], true)
        if a.noise:
            surf = Surface(surf.S, surf.r, surf.q, surf.K, surf.T, surf.iv + np.random.default_rng(a.seed).normal(0, a.noise, len(surf.iv)))
        t0 = time.time()
        res = calibrate(surf)
        print(f"true      v0={true.v0:.4f} kappa={true.kappa:.3f} theta={true.theta:.4f} sigma_v={true.sigma_v:.3f} rho={true.rho:.3f}")
        print(f"fitted    {res.summary()}")
        print(f"{res.n_points} points, {len(res.starts)} starts, {time.time() - t0:.1f} s")
        for t in res.starts:
            s0 = t["start"]
            print(f"  from v0={s0.v0:.2f} kappa={s0.kappa:.1f} theta={s0.theta:.2f} sigma_v={s0.sigma_v:.1f} rho={s0.rho:.1f}: "
                  f"RMSE {t['rmse_vol'] * 100:.4f} vol pts, kappa {t['params'].kappa:.3f}")
    elif a.cmd == "price":
        print(_price(a))


def _price(a) -> str:
    from . import bs, fd, heston, mc, trees

    ex = "american" if a.american else "european"
    S, K, T, s, r, q, right = a.S, a.K, a.T, a.sigma, a.r, a.q, a.right
    if a.method == "bs":
        if a.american:
            raise SystemExit("bs is European only")
        g = bs.greeks(S, K, T, s, right, r, q)
        return (f"{g.price:.6f}  delta {g.delta:.6f} gamma {g.gamma:.6f} vega {g.vega:.6f} "
                f"theta/day {g.theta:.6f} rho {g.rho:.6f}")
    if a.method in ("crr", "jr"):
        res = trees.binomial(S, K, T, s, right, a.n or 1000, r, q, exercise=ex, method=a.method)
        return f"{res.price:.6f}  ({res.method} N={res.n} {res.exercise})"
    if a.method == "trinomial":
        res = trees.trinomial(S, K, T, s, right, a.n or 500, r, q, exercise=ex)
        return f"{res.price:.6f}  ({res.method} N={res.n} {res.exercise})"
    if a.method.startswith("fd-"):
        scheme = a.method[3:]
        solver = "bs" if right == "P" else "psor"
        res = fd.price(S, K, T, s, right, r, q, n_x=a.n or 400, scheme=scheme, exercise=ex, american=solver)
        tag = f" {res.american_solver}" if res.american_solver else ""
        return f"{res.price:.6f}  (fd {res.scheme} n_x={res.n_x} n_t={res.n_t} {res.exercise}{tag})"
    if a.method in ("mc", "mc-cv"):
        if a.american:
            raise SystemExit("use --method lsm for an American put by Monte Carlo")
        cv = a.method == "mc-cv"
        res = mc.european(S, K, T, s, right, r, q, n=a.n or 100_000, seed=a.seed, antithetic=cv, control_variate=cv)
        return f"{res.price:.6f}  se {res.se:.6f}  95% CI [{res.ci_lo:.6f}, {res.ci_hi:.6f}]  ({res.variant} n={res.n})"
    if a.method == "lsm":
        if right != "P":
            raise SystemExit("lsm is the American put")
        res = mc.american_put_lsm(S, K, T, s, r, q, n=a.n or 100_000, seed=a.seed)
        return f"{res.price:.6f}  se {res.se:.6f}  (lsm n={res.n}, 50 exercise dates/yr)"
    if a.method == "cos-gbm":
        if a.american:
            raise SystemExit("cos-gbm is European only")
        return f"{heston.price_gbm(S, K, T, s, right, r, q, N=a.n or 256):.6f}  (COS, GBM cf)"
    raise SystemExit(f"unknown method {a.method}")
