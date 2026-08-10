# Project: Markov-Switching TVTP Monte Carlo Study

## Architecture
```
project/
├── config/          # Settings and DGP parameter definitions
│   ├── settings.py  # R, N_STARTS, T_LIST, BURN_IN, END_TRIM, SEED_BASE, etc.
│   └── dgp_params.py # DGP dataclass + get_all_dgps() returning dict[int, DGP]
├── data/
│   └── treasury.py  # load_treasury_yields(maturities) -> pd.DataFrame
├── src/
│   ├── filter.py    # build_transition_matrix, hamilton_filter
│   ├── simulate.py  # simulate_dgp(dgp, T, rng) -> (y, z, X)
│   ├── estimate.py  # estimate_model, pack_params, unpack_params, correct_label_switching, neg_loglik
│   ├── metrics.py   # compute_parameter_metrics, compute_forecast_metrics, compute_filtered_prob_errors
│   └── reporting.py # aggregate_mc_results, format_results_table
├── exp/
│   ├── exp1_monte_carlo.py  # Main MC: 9 DGPs × T∈{500,1000} × R=50 reps
│   ├── exp2_gas_diagnostic.py # GAS non-identifiability diagnostic (DGPs 4, 9)
│   └── exp3_empirical.py    # Treasury yield K=3 empirical application
├── tests/
│   ├── test_filter.py
│   ├── test_simulate.py
│   ├── test_metrics.py
│   └── test_estimate.py
└── results/
    ├── raw/         # Per-DGP pickled replication results
    ├── aggregated/  # Per-DGP CSV summaries + master CSV
    └── figures/     # PNG plots (regime classification, GAS profiles)
```

## DGP Index (9 DGPs)
| ID | K | Transition model  | Param.       | Common σ² | Has X |
|----|---|-------------------|--------------|-----------|-------|
| 1  | 2 | Constant          | diagonal     | True      | No    |
| 2  | 2 | ModelI (TVTP)     | diagonal     | True      | No    |
| 3  | 2 | ModelII (exog.)   | diagonal     | True      | Yes   |
| 4  | 2 | ModelIII (GAS)    | diagonal     | True      | No    |
| 5  | 2 | ModelI (TVTP)     | off_diagonal | False     | No    |
| 6  | 3 | Constant          | off_diagonal | False     | No    |
| 7  | 3 | ModelI (extreme A)| off_diagonal | False     | No    |
| 8  | 3 | ModelII (exog.)   | off_diagonal | False     | Yes   |
| 9  | 3 | ModelIII (GAS)    | off_diagonal | False     | No    |

## Key Settings (config/settings.py)
- R=50 replications, SEED_BASE=12345
- T_LIST=[500, 1000], BURN_IN=100, END_TRIM=10
- Effective obs: 390 (T=500), 890 (T=1000)
- N_STARTS=10 (MC), N_STARTS_EMP=20 (empirical)
- EXTREME_A_THRESHOLD=10.0 (DGP7 filter)
- EFF_OBS_EMP=662 (empirical: 762 – 100 burn-in)

## DGP Dataclass Fields
- dgp_id: int
- K: int (2 or 3)
- mu: List[float] — regime means
- sigma2: List[float] — [single] if common_variance else [K values]
- common_variance: bool
- parameterization: str — 'diagonal' | 'off_diagonal'
- transition_model: str — 'Constant' | 'ModelI' | 'ModelII' | 'ModelIII'
- omega: List[float] — logit-space baseline transition params
- A: Optional[List[float]] — score/exogenous coefficient (None for Constant)
- B: Optional[List[float]] — persistence coefficient (None for Constant/ModelII)
- estimation_models: List[str] — models estimated for this DGP

## omega Parameterization Conventions
- diagonal (K=2): omega = [logit(p_00), logit(p_11)]  → 2 params
- off_diagonal (K=3): omega = [logit(p_01), logit(p_02),  → 6 params
                                logit(p_10), logit(p_12),
                                logit(p_20), logit(p_21)]
  Row-wise off-diagonal elements; diagonal from 1 − row_sum (clipped/rescaled)

## Transition Model Conventions
- Constant: f_t = omega (fixed)
- ModelI:   f_t = omega + A * score_t-1 + B * (f_t-1 − omega)
- ModelII:  f_t = omega + A * X_t  (X is exogenous)
- ModelIII (GAS): f_t = omega + A * score_t-1 + B * (f_t-1 − omega)
  (same recursion as ModelI but A is non-identifiable → should ≈ 0)

## Filter Output Convention
hamilton_filter returns (loglik: float, filtered_probs: (T,K), pred_probs: (T,K))
- filtered_probs[t] = P(s_t | y_{1:t})
- pred_probs[t]     = P(s_t | y_{1:t-1})  (one-step-ahead)

## Label Switching
correct_label_switching permutes regime indices so estimated mu aligns with
true mu by minimising sum of absolute deviations. Permutes mu, sigma2, omega,
A, B, filtered_probs, pred_probs consistently.

## DGP7 Special Case
At T=500, ModelI: some replications produce |A| > EXTREME_A_THRESHOLD.
These are excluded from the A-parameter summary statistics only (not from
convergence counts or other metrics). See exp1 converged_for_A logic.

## Empirical Application (exp3)
- 763 monthly Treasury yield observations (differenced → 762 changes)
- Maturities: 1m, 12m, 36m, 72m
- K=3, off_diagonal, common_variance=False
- BURN_IN=100, END_TRIM=0 for empirical (EFF_SAMPLE=662)
- X = lagged level for ModelII

## Implementation Status
All core-infrastructure files are fully implemented (no stubs/TODOs):
- src/filter.py:    Hamilton filter + TVTP (Constant/ModelI/II/III-GAS), K=2 diagonal,
                    K=2 off-diagonal, K=3 off-diagonal with row rescaling, OPG EMA score
- src/simulate.py:  All 9 DGPs; GAS simulation uses same score recursion as filter
- src/estimate.py:  Multistart MLE (L-BFGS-B + Nelder-Mead fallback), pack/unpack,
                    numerical Hessian, SE, label-switching correction
- src/metrics.py:   Bias/RMSE/Coverage95, MAFE/MSFE, filtered-prob MSE/MAE
- src/reporting.py: MC aggregation, Markdown + LaTeX table formatters
- data/treasury.py: 3-source loader (cache → GitHub → synthetic fallback AR(1) RS-vol)

## Quick Sanity Check
```bash
cd /workspace/project
python -c "from src.filter import hamilton_filter, build_transition_matrix; print('Filter OK')"
```

## Key Implementation Details
- Parameter vector layout: [mu(K), log_sigma2(1 or K), omega(n_t), A(n_t), B(n_t)]
  where n_t = K*(K-1) for off_diagonal, K for diagonal
- GAS score: closed-form d/d f_t of log predictive density; Fisher = OPG EMA (rho=0.95)
- K=3 off-diagonal row rescaling triggers when sum(sigmoid(f_row)) > 1 - 1e-10
- Likelihood window: t in [burn_in, T-end_trim), giving 390/890 effective obs in sim
- Treasury fallback: 763-obs AR(1) level + slope + regime-switching vol, 4 maturities

## Testing Notes
- Run: cd /workspace/project && python -m pytest tests/ -v --tb=short
- All functional sanity checks pass (see implementation session verification above)
