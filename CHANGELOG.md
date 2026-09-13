# Changelog

## 0.2.0 — 2026-09-12
- `calibrate`: Heston calibration by least squares in implied-vol space with bounds and multi-start; `CalibrationResult` carries RMSE, max residual, Feller flag and every start's outcome. Recovers generating parameters from three starts; fits 20 bp noise to the noise level.
- `heston.price_strikes`: one characteristic-function evaluation per maturity for all strikes (equal to `price` to 1e-12).
- `bs.implied_vol_vec`: vectorised Newton in sigma with bisection guard and Brent fallback; agrees with Brent to 1e-8; NaN for prices within 1e-10·S of intrinsic (uninformative), documented.
- `pricers calibrate` CLI demo.

## 0.1.1 — 2026-09-12
- `heston` — COS truncation range now placed at ln(S/K) + c1 with the put support [a, min(b, 0)] (v0.1 placed it at c1: exact at the money, wrong beyond about 4.5 sigma sqrt(T), e.g. a K = 200, T = 0.02 GBM call priced at -99.8); default N raised from 256 to 1024 (Feller-violated, high vol-of-vol parameters left 2e-3 errors at N = 256; 0 of 400 random cases above 1e-6 vs QuantLib at 1024); tests at 6-10 sd, a Feller-violated case, the COS convergence rate, the c2 check tightened to 1e-9 and the paper's printed c2 gap asserted
- `fd` — the step is exactly half-width / (n_x / 2); the v0.1 nudge that put S on a node pinned h near the money and stalled refinement (ratio 1.00 between n_x = 400 and 800 at S = 100.373); a test keeps the off-node second-order ratio
- `trees` — American intrinsic values from a precomputed (u/d)^j, 0.37 s -> 55 ms CPU at N = 5,000
- `mc` — the LSM standard error is documented as the paper's conditional one (understates by up to ~16% with a large exercise region, measured over 150 seeds)
- docs: trees timing and BBSR gain corrected (6.1e-3, 19x), README section order per the sibling convention, report row for Jarrow-Rudd split from the 2.2/N rows, CI installs `[dev]` and `[oracle]` in separate steps

## 0.1.0 — 2026-09-12
- `bs` — Black-Scholes-Merton price, Greeks, implied vol (Brent); the deskboard API verbatim, plus array inputs
- `trees` — CRR (exact and QuantLib-form p), Jarrow-Rudd, Kamrad-Ritchken trinomial; European and American; BBS smoothing and Richardson extrapolation; vectorised backward induction
- `fd` — log-spot grid, explicit / implicit / Crank-Nicolson with Rannacher start-up; American by PSOR and by Brennan-Schwartz projected Thomas (puts); bump Greeks
- `mc` — GBM terminal and path simulation; plain, antithetic, S_T control variate (optimal beta) and both; Longstaff-Schwartz American put
- `heston` — COS method with the Albrecher/Gatheral characteristic function; truncation range at ln(S/K) + c1 with closed-form c1, c2 (the paper's printed c2 corrected) and numeric c4; N = 1024 by default
- `pricers bench`, `pricers report`, `pricers price`; oracle fixture with sources; QuantLib as an optional `[oracle]` dependency; CI on 3.11 / 3.12 with a zero-spend reproducibility step
