# Markov-Switching TVTP Monte Carlo Study

A from-scratch Python implementation of a Monte Carlo simulation study for
**Markov-Switching models with Time-Varying Transition Probabilities (TVTP)**,
covering diagonal and off-diagonal parameterisations, GAS (Generalised
Autoregressive Score) dynamics, and empirical application to U.S. Treasury yields.

No mainstream Python library handles TVTP-GAS Markov switching with
off-diagonal row rescaling and the exact Bazzi et al. (2017) likelihood
trimming conventions — this codebase fills that gap.

---

## Directory layout

```
.
├── config/
│   ├── settings.py      # Global MC settings (R, T_LIST, burn-in, …)
│   └── dgp_params.py    # All 9 DGP configurations as dataclasses
├── src/
│   ├── filter.py        # Hamilton filter (Constant / ModelI / II / III-GAS)
│   ├── simulate.py      # DGP simulation for all 9 processes
│   ├── estimate.py      # Multistart MLE with label-switching correction
│   ├── metrics.py       # Bias, RMSE, coverage, MAFE, MSFE, filtered-prob errors
│   └── reporting.py     # MC aggregation, Markdown and LaTeX table formatters
├── data/
│   ├── treasury.py      # Liu & Wu (2021) Treasury yield loader (w/ fallback)
│   └── raw/             # Local CSV cache (git-ignored)
├── results/
│   ├── raw/             # Per-replication pickle files
│   ├── aggregated/      # Summary DataFrames / CSVs
│   └── figures/         # Plots
├── requirements.txt
└── README.md
```

---

## Models

### Transition specifications

| Label    | Logit equation                                      |
|----------|-----------------------------------------------------|
| Constant | f_t = ω                                            |
| ModelI   | f_t = ω + A · y_{t-1}                              |
| ModelII  | f_t = ω + A · X_{t-1}  (X exogenous)              |
| ModelIII | f_t = ω + A · s_{t-1} + B · (f_{t-1} − ω)  (GAS) |

### DGPs

| ID | K | Param.       | Variance | True model  |
|----|---|--------------|----------|-------------|
| 1  | 2 | diagonal     | common   | Constant    |
| 2  | 2 | diagonal     | common   | ModelI      |
| 3  | 2 | diagonal     | common   | ModelII     |
| 4  | 2 | diagonal     | common   | ModelIII    |
| 5  | 2 | off-diagonal | specific | ModelI      |
| 6  | 3 | off-diagonal | specific | Constant    |
| 7  | 3 | off-diagonal | specific | ModelI      |
| 8  | 3 | off-diagonal | specific | ModelII     |
| 9  | 3 | off-diagonal | specific | ModelIII    |

---

## Key design choices

* **Off-diagonal row rescaling** (K=3): when sigmoid-mapped off-diagonal
  probabilities in a row exceed 1, they are rescaled proportionally so that
  the row sums to 1 and the diagonal remains non-negative.

* **GAS score** (ModelIII): closed-form derivative of the predictive
  log-likelihood w.r.t. each logit parameter.  Fisher scaling uses a
  recursive OPG EMA with decay ρ = 0.95.

* **Likelihood window**: accumulation over t ∈ [burn_in, T − end_trim),
  matching the paper's effective observation counts (390 / 890 in simulation;
  662 in the empirical application).

* **Label-switching correction**: after each MLE run, regimes are permuted
  to minimise ∑_i |μ̂_{π(i)} − μ_i^{true}|.

---

## Quick start

```python
import numpy as np
from config.dgp_params import DGP2
from src.simulate import simulate_dgp
from src.estimate import estimate_model

rng = np.random.default_rng(42)
y, z, X = simulate_dgp(DGP2, T=500, rng=rng)
result = estimate_model(
    y, K=2, parameterization='diagonal', model='ModelI',
    common_variance=True, n_starts=10,
    burn_in=100, end_trim=10, rng=rng,
)
print(f"Log-lik: {result['loglik']:.2f}")
print(f"mu_hat:  {result['params']['mu']}")
```

---

## References

* Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series and the business cycle.* Econometrica, 57(2), 357–384.
* Bazzi, M., Blasques, F., Koopman, S. J., & Lucas, A. (2017). *Time-varying
  transition probabilities for Markov regime switching models.* Journal of
  Time Series Analysis, 38(3), 458–478.
* Liu, Y., & Wu, J. C. (2021). *Reconstructing the yield curve.*
  Journal of Financial Economics, 142(3), 1395–1425.
