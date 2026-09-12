"""`pricers report`: regenerate the README's validation table (method, oracle, tolerance, cases,
measured max |error|, status) and convergence-rate table from the oracle fixture, and write them
between `<!-- report:start -->` / `<!-- report:end -->`. Every number in those two tables comes
from here; the tests assert the same quantities against the same tolerances. The output is
deterministic (fixed seeds, errors printed to two significant figures), so CI regenerates it and
requires `git diff --exit-code`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bs, fd, heston, mc, trees

START, END = "<!-- report:start -->", "<!-- report:end -->"
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "oracle_values.json"
CANON = (100.0, 100.0, 1.0, 0.2, 0.05)   # S, K, T, sigma, r


@dataclass(frozen=True)
class Check:
    method: str
    oracle: str
    tol: float
    errors: list[float]
    unit: str = ""          # "" for price units, "s.e." when the errors are in standard errors

    @property
    def worst(self) -> float:
        return max(self.errors)

    @property
    def status(self) -> str:
        return "pass" if self.worst <= self.tol else "FAIL"


def _ref_call_put(right):
    S, K, T, s, r = CANON
    return bs.price(S, K, T, s, right, r)


def validation(o: dict) -> list[Check]:
    S, K, T, s, r = CANON
    out: list[Check] = []
    c = o["bs_european"]["canonical"]
    out.append(Check("Black-Scholes closed form", "S=K=100, r=5%, sigma=20%, T=1 (scipy, QuantLib)", c["tol"], [
        abs(bs.price(S, K, T, s, "C", r) - c["call"]), abs(bs.price(S, K, T, s, "P", r) - c["put"]),
        abs(bs.price(S, K, T, s, "C", r) - bs.price(S, K, T, s, "P", r) - c["parity_c_minus_p"])]))
    h = o["bs_european"]["hull_example"]
    out.append(Check("Black-Scholes closed form", "Hull worked example 4.76 / 0.81 (S=42, K=40, r=10%, sigma=20%, T=0.5)",
                     h["tol_vs_hull"], [abs(bs.price(h["S"], h["K"], h["T"], h["sigma"], "C", h["r"]) - h["hull_rounded"]["call"]),
                                        abs(bs.price(h["S"], h["K"], h["T"], h["sigma"], "P", h["r"]) - h["hull_rounded"]["put"])]))
    g = o["bs_european"]["fang_oosterlee_gbm"]
    out.append(Check("Black-Scholes closed form", "Fang & Oosterlee 2008 Table 2 (GBM calls, three strikes)", g["tol"],
                     [abs(bs.price(g["S"], float(k), g["T"], g["sigma"], "C", g["r"]) - v) for k, v in g["calls"].items()]))
    out.append(Check("COS, GBM characteristic function", "Black-Scholes and Fang & Oosterlee Table 2", 1e-8,
                     [abs(heston.price_gbm(g["S"], float(k), g["T"], g["sigma"], "C", g["r"]) - v) for k, v in g["calls"].items()]
                     + [abs(heston.price_gbm(x, K, T, s, rt, r, 0.01) - bs.price(x, K, T, s, rt, r, 0.01))
                        for rt in "CP" for x in (80.0, 100.0, 125.0)]))
    out.append(Check("CRR and Kamrad-Ritchken trinomial trees, European, N=2000", "Black-Scholes; bound 2.2/N", 2.2 / 2000, [
        abs(trees.binomial(S, K, T, s, rt, 2000, r).price - _ref_call_put(rt)) for rt in "CP"]
        + [abs(trees.trinomial(S, K, T, s, rt, 2000, r).price - _ref_call_put(rt)) for rt in "CP"]))
    out.append(Check("Jarrow-Rudd tree, European, N=2000", "Black-Scholes (no pre-registered bound; test bar 2e-3)", 2e-3, [
        abs(trees.binomial(S, K, T, s, rt, 2000, r, method="jr").price - _ref_call_put(rt)) for rt in "CP"]))
    hp = o["hull_american_put_tree"]
    out.append(Check("CRR American put, 5 steps", "Hull 5-step tree 4.49", 5e-3, [
        abs(trees.binomial(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", 5, hp["r"], exercise="american").price
            - hp["hull_rounded_5_steps"])]))
    out.append(Check("CRR / JR (N=5000), KR trinomial (N=2000), American put", "Hull put converged 4.2842 (QuantLib QdFp)", 1e-3, [
        abs(trees.binomial(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", 5000, hp["r"], exercise="american", method=m).price
            - hp["converged"]) for m in ("crr", "jr")]
        + [abs(trees.trinomial(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", 2000, hp["r"], exercise="american").price
               - hp["converged"])]))
    out.append(Check("BBSR (BBS + Richardson, N=100/200), American put", "Hull put converged 4.2842", 5e-4, [
        abs(trees.richardson(lambda n: trees.binomial(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", n, hp["r"],
                                                      exercise="american", bbs=True).price, 100) - hp["converged"])]))
    out.append(Check("FD Crank-Nicolson + Rannacher, European, n_x=n_t=800", "Black-Scholes (S on and off the strike node)", 2e-4, [
        abs(fd.price(x, K, T, s, rt, r, 0.01, n_x=800).price - bs.price(x, K, T, s, rt, r, 0.01))
        for x, rt in ((90.0, "C"), (95.3, "C"), (100.0, "P"), (117.2, "P"))]))
    lsm = o["american_put_lsm2001_table1"]
    rows = lsm["rows"]
    bs_am = [fd.price(w["S"], lsm["K"], w["T"], w["sigma"], "P", lsm["r"], n_x=800, exercise="american", american="bs").price
             for w in rows]
    out.append(Check("FD CN + Brennan-Schwartz, American put, 800 x 800", "Longstaff-Schwartz Table 1, continuous-American column (QuantLib QdFp)",
                     lsm["tolerances"]["american_continuous_vs_converged_psor"],
                     [abs(p - w["american_continuous"]) for p, w in zip(bs_am, rows, strict=True)]))
    out.append(Check("FD CN + PSOR, American put, 200 x 200 (3 cases)", "same column", 5e-3, [
        abs(fd.price(w["S"], lsm["K"], w["T"], w["sigma"], "P", lsm["r"], n_x=200, exercise="american", american="psor").price
            - w["american_continuous"]) for w in (rows[0], rows[10], rows[19])]))
    out.append(Check("FD PSOR vs Brennan-Schwartz, same 150 x 150 grid", "each other (Hull put)", 1e-6, [abs(
        fd.price(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", hp["r"], n_x=150, exercise="american", american="psor").price
        - fd.price(hp["S"], hp["K"], hp["T"], hp["sigma"], "P", hp["r"], n_x=150, exercise="american", american="bs").price)]))
    z = []
    for seed in range(4):
        for av in (False, True):
            for cv in (False, True):
                for rt in "CP":
                    res = mc.european(S, K, T, s, rt, r, n=200_000, seed=seed, antithetic=av, control_variate=cv)
                    z.append(abs(res.price - _ref_call_put(rt)) / res.se)
    out.append(Check("MC plain / antithetic / control variate / both, n=200,000, seeds 0-3", "Black-Scholes", 3.0, z, "s.e."))
    lsm_z, lsm_d = [], []
    for w in rows:
        res = mc.american_put_lsm(w["S"], lsm["K"], w["T"], w["sigma"], lsm["r"], n=100_000, seed=1)
        lsm_z.append(abs(res.price - w["lsm_sim"]) / float(np.hypot(res.se, w["lsm_se"])))
        lsm_d.append(abs(res.price - w["american_continuous"]))
    out.append(Check("LSMC American put, 100,000 antithetic paths, 50 dates/yr, seed 1", "Longstaff-Schwartz Table 1 LSM column (combined s.e.)", 3.0, lsm_z, "s.e."))
    out.append(Check("LSMC American put (same run)", "continuous-American column", 0.03, lsm_d))
    fo = o["heston"]["fang_oosterlee_2008"]
    hpar = heston.HestonParams(fo["v0"], fo["kappa"], fo["theta"], fo["sigma_v"], fo["rho"])
    out.append(Check("Heston COS, N=1024, L=12, numeric c4", "Fang & Oosterlee 2008 eq. (53), T=1 and T=10 (Feller violated)", 1e-6, [
        abs(heston.price(fo["S"], fo["K"], 1.0, "C", hpar) - fo["call_T1"]["published"]),
        abs(heston.price(fo["S"], fo["K"], 10.0, "C", hpar) - fo["call_T10"]["published"])]))
    an = o["heston"]["andersen_2008_case1"]
    apar = heston.HestonParams(an["v0"], an["kappa"], an["theta"], an["sigma_v"], an["rho"])
    out.append(Check("Heston COS, N=2048, L=30, c4=0", "Andersen 2008 Case I, T=10 (sigma_v = 1; QuantLib analytic)", an["tol"], [
        abs(heston.price(an["S"], an["K"], an["T"], "C", apar, N=2048, L=30.0, c4=0) - an["call"])]))
    return out


def convergence(o: dict) -> list[tuple[str, str, str]]:
    S, K, T, s, r = CANON
    ref = _ref_call_put("C")
    c = o["convergence_constants_S100_K100_r5_sig20_T1_call"]
    crr = lambda n, m="crr": (trees.binomial(S, K, T, s, "C", n, r, method=m).price - ref) * n  # noqa: E731
    cn = [abs(fd.price(S, K, T, s, "C", r, n_x=n).price - ref) for n in (50, 100, 200, 400, 800)]
    im = [abs(fd.price(S, K, T, s, "C", r, n_x=n, scheme="implicit").price - ref) for n in (100, 200, 400)]
    sd = c["mc_exact_sd_discounted_payoff"]
    plain = mc.european(S, K, T, s, "C", r, n=200_000, seed=7)
    plain_p = mc.european(S, K, T, s, "P", r, n=200_000, seed=7)
    av = mc.european(S, K, T, s, "C", r, n=200_000, seed=7, antithetic=True)
    cv = mc.european(S, K, T, s, "C", r, n=200_000, seed=7, control_variate=True)
    fo = o["heston"]["fang_oosterlee_2008"]
    hpar = heston.HestonParams(fo["v0"], fo["kappa"], fo["theta"], fo["sigma_v"], fo["rho"])
    cos = [abs(heston.price(100.0, 100.0, 1.0, "C", hpar, N=n) - fo["call_T1"]["published"]) for n in (32, 64, 128, 256)]
    rows = [
        ("CRR N x error, N=800 / 801, exact p (Hull's form)", f"{crr(800):+.3f} / {crr(801):+.3f}",
         "-2.000 / +1.753 measured; bound abs(error) <= 2.2/N"),
        ("CRR N x error, N=800 / 801, QuantLib's first-order p", f"{crr(800, 'crr-ql'):+.3f} / {crr(801, 'crr-ql'):+.3f}",
         f"{c['crr_N_times_error']['even_N']:+.3f} / {c['crr_N_times_error']['odd_N']:+.3f} (fixture)"),
        ("CRR error(800) / error(400), same parity", f"{crr(800) / 800 / (crr(400) / 400):.3f}", "0.5 (first order)"),
        ("Crank-Nicolson + Rannacher error ratio on doubling, n=50..800", " / ".join(f"{a / b:.2f}" for a, b in zip(cn[:-1], cn[1:], strict=True)),
         "4.0 (second order; fixture: QuantLib 4.18, 4.09, 4.04, 4.02)"),
        ("Implicit Euler error ratio on doubling, n=100..400", " / ".join(f"{a / b:.2f}" for a, b in zip(im[:-1], im[1:], strict=True)),
         "between 2 (first order in dt) and 4 (second order in h)"),
        ("MC plain: s.e. x sqrt(n), call / put, n=200,000", f"{plain.se * np.sqrt(plain.n):.3f} / {plain_p.se * np.sqrt(plain_p.n):.3f}",
         f"{sd['call']:.3f} / {sd['put']:.3f} exact sd of the discounted payoff"),
        ("MC antithetic: s.e. x sqrt(n), call", f"{av.se * np.sqrt(av.n):.2f}", f"{sd['antithetic_equiv_per_draw_sd_call']:.2f} (fixture)"),
        ("MC S_T control variate: variance factor, beta", f"{cv.variance_factor:.4f}, {cv.beta:.3f}",
         f"{sd['control_variate_S_T']['variance_factor']:.4f} = 1 - corr^2, corr {sd['control_variate_S_T']['corr']:.4f}"),
        ("Heston COS abs(error) at N = 32 / 64 / 128 / 256, T=1", " / ".join(f"{e:.1e}" for e in cos),
         "exponential in N: each doubling > 10x (tested)"),
    ]
    return rows


def tables(o: dict | None = None) -> str:
    o = o or json.loads(FIXTURE.read_text())
    lines = ["Validation (regenerated by `pricers report`; the tests assert the same quantities):", "",
             "| method | oracle | tolerance | cases | max abs error | status |", "|---|---|---:|---:|---:|---|"]
    for ch in validation(o):
        tol = f"{ch.tol:g} {ch.unit}".strip()
        worst = f"{ch.worst:.2f} {ch.unit}" if ch.unit else f"{ch.worst:.1e}"
        lines.append(f"| {ch.method} | {ch.oracle} | {tol} | {len(ch.errors)} | {worst} | {ch.status} |")
    lines += ["", "Convergence rates (regenerated by `pricers report`):", "",
              "| quantity | measured | expected |", "|---|---|---|"]
    for q, m, e in convergence(o):
        lines.append(f"| {q} | {m} | {e} |")
    return "\n".join(lines)


def write_readme(readme: Path, text: str) -> None:
    src = readme.read_text()
    if START not in src or END not in src:
        raise SystemExit(f"{readme}: markers {START} / {END} not found")
    head, rest = src.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{text}\n{END}{tail}")
