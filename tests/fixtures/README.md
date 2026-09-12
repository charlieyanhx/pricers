# Oracle fixture

`oracle_values.json` is the reference-value file the oracle tests read. It is kept verbatim
(no keys added) so that its provenance is the file itself; every value carries a `source`
field and a `tol` field, and the tests assert against those tolerances. Sources, in the order
they appear:

- `bs_european.canonical` — S = K = 100, r = 5%, sigma = 20%, T = 1: call 10.4505835722, put
  5.5735260223. Closed form recomputed with scipy and QuantLib 1.43 `AnalyticEuropeanEngine`.
  This is not a textbook example.
- `bs_european.hull_example` — S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5: Hull, *Options,
  Futures, and Other Derivatives*, Black-Scholes-Merton chapter worked example (4.76 / 0.81).
- `bs_european.fang_oosterlee_gbm` and `fang_oosterlee_cash_or_nothing` — Fang & Oosterlee
  (2008), *A novel pricing method for European options based on Fourier-cosine series
  expansions*, SIAM J. Sci. Comput. 31(2), eqs. (51)-(52), Tables 2-3.
- `american_put_lsm2001_table1` — Longstaff & Schwartz (2001), *Valuing American options by
  simulation: a simple least-squares approach*, RFS 14(1), Table 1: the paper's finite-difference
  column (a Bermudan put exercisable 50 times a year), its LSM column with standard errors, the
  European values, and a continuous-American column computed with QuantLib 1.43
  `QdFpAmericanEngine` (high precision), cross-checked against `FdBlackScholesVanillaEngine`
  (2000 x 2000) and CRR with N = 5000 to 4e-4.
- `hull_american_put_tree` — Hull, *Basic Numerical Procedures* chapter: the 5-step CRR
  American put (S = K = 50, r = 10%, sigma = 40%, T = 5/12) = 4.49; converged value from QuantLib
  `QdFpAmericanEngine` 4.28422 and FD 4.28408.
- `heston.fang_oosterlee_2008` — Fang & Oosterlee (2008) eq. (53), Tables 4-5, parameters from
  Albrecher, Mayer, Schoutens & Tistaert (2007), *The little Heston trap* (Feller condition
  violated); the `quantlib_5_engines` key is a QuantLib cross-check recorded at value-collection
  time (the engines are not named in the file; the test here re-checks against `COSHestonEngine`
  and `AnalyticHestonEngine`).
- `heston.andersen_2008_case1` — Andersen (2008), *Simple and efficient simulation of the Heston
  stochastic volatility model*, J. Comp. Finance, Case I (attribution noted in the file as from
  memory); value verified with QuantLib `AnalyticHestonEngine`.
- `convergence_constants_S100_K100_r5_sig20_T1_call` — measured on the canonical case: CRR
  N x error limits (QuantLib's CRR, whose branch probability is the first-order
  1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma); the exact-p textbook CRR in `pricers.trees`
  gives -2.000 / +1.753 and both satisfy |error| <= 2.2 / N), Crank-Nicolson error ratios on grid
  doubling with the strike on a node, and the exact standard deviation of the discounted payoff
  with the S_T control-variate variance factor.
