# Changelog

## 0.1.0 — 2026-09-11
- `bs` — Black-Scholes-Merton price, Greeks, implied vol (Brent); the deskboard API verbatim, plus array inputs
- `trees` — CRR (exact and QuantLib-form p), Jarrow-Rudd, Kamrad-Ritchken trinomial; European and American; BBS smoothing and Richardson extrapolation; vectorised backward induction
- `fd` — log-spot grid, explicit / implicit / Crank-Nicolson with Rannacher start-up; American by PSOR and by Brennan-Schwartz projected Thomas (puts); bump Greeks
- `mc` — GBM terminal and path simulation; plain, antithetic, S_T control variate (optimal beta) and both; Longstaff-Schwartz American put
- `heston` — COS method with the Albrecher/Gatheral characteristic function; closed-form c1, c2 (the paper's printed c2 corrected), numeric c4 in the truncation range
- `pricers bench`, `pricers report`, `pricers price`; oracle fixture with sources; QuantLib as an optional `[oracle]` dependency; CI on 3.11 / 3.12 with a zero-spend reproducibility step
