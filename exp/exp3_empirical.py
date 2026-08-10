"""
Experiment 3: Empirical Markov-Switching on U.S. Treasury Yield Changes.
K=3 models at maturities 1m, 12m, 36m, 72m.
Models: Constant, ModelI (TVP), ModelII (Exogenous level), attempted ModelIII (GAS).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BURN_IN, EFF_OBS_EMP, N_STARTS_EMP
from data.treasury import load_treasury_yields
from src.filter import hamilton_filter
from src.estimate import estimate_model

MATURITIES = [1, 12, 36, 72]   # in months
MODELS = ['Constant', 'ModelI', 'ModelII', 'ModelIII']
K = 3
EFF_SAMPLE = 662   # 762 obs after differencing, minus 100 burn-in

FIGURES_DIR = Path(__file__).parent.parent / 'results' / 'figures'
RESULTS_AGG_DIR = Path(__file__).parent.parent / 'results' / 'aggregated'


# ---------------------------------------------------------------------------
# Helper: parameter count
# ---------------------------------------------------------------------------

def count_params(
    K: int,
    parameterization: str,
    model: str,
    common_variance: bool,
) -> int:
    """Count free parameters for BIC/AIC computation."""
    n = K                                  # mu
    n += 1 if common_variance else K      # sigma2
    if parameterization == 'diagonal':
        n_omega = K
    else:                                  # off_diagonal
        n_omega = K * (K - 1)
    n += n_omega                          # omega (baseline transition)
    if model in ('ModelI', 'ModelIII'):
        n += n_omega                       # A
        n += n_omega                       # B
    elif model == 'ModelII':
        n += n_omega                       # A (exogenous coefficients)
    return n


# ---------------------------------------------------------------------------
# Helper: extract differenced yield series for one maturity
# ---------------------------------------------------------------------------

def y_diff_for_mat(yields: pd.DataFrame, mat: int) -> np.ndarray:
    """Return the first-differenced yield series for maturity `mat`."""
    col = f'{mat}m'
    return np.diff(yields[col].values)


# ---------------------------------------------------------------------------
# Main estimation loop
# ---------------------------------------------------------------------------

def estimate_all_maturities(
    n_starts: int = N_STARTS_EMP,
) -> pd.DataFrame:
    """
    Estimate all models for all maturities.

    Returns DataFrame with columns:
    Maturity, Model, LogLik, nParams, AIC, BIC, Converged, MaxAbsA
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AGG_DIR.mkdir(parents=True, exist_ok=True)

    yields = load_treasury_yields(MATURITIES)

    rows = []
    all_results: Dict = {}

    for mat in MATURITIES:
        col = f'{mat}m'
        level_series = yields[col].values        # shape (N,)
        y = np.diff(level_series)                # shape (N-1,) — yield changes
        X_level = level_series[:-1]              # shape (N-1,) — lagged levels

        for model in MODELS:
            print(f'Estimating maturity={mat}m, model={model}...')

            X = X_level if model == 'ModelII' else None

            result = estimate_model(
                y, K, 'off_diagonal', model,
                common_variance=False,           # K=3 regime-specific variance
                n_starts=n_starts,
                burn_in=BURN_IN,
                end_trim=0,                      # no end_trim for empirical
                rng=np.random.default_rng(42),
                X=X,
            )

            n_params = count_params(K, 'off_diagonal', model, common_variance=False)
            ll = result['loglik']
            if not np.isfinite(ll):
                ll = np.nan
            aic = (2 * n_params - 2 * ll) if np.isfinite(ll) else np.nan
            bic = (n_params * np.log(EFF_SAMPLE) - 2 * ll) if np.isfinite(ll) else np.nan

            max_abs_A = None
            if result['params'].get('A') is not None:
                max_abs_A = round(float(np.max(np.abs(result['params']['A']))), 4)

            all_results[(mat, model)] = result
            rows.append({
                'Maturity': f'{mat}m',
                'Model': model,
                'LogLik': round(ll, 2) if np.isfinite(ll) else np.nan,
                'nParams': n_params,
                'AIC': round(aic, 2) if np.isfinite(aic) else np.nan,
                'BIC': round(bic, 2) if np.isfinite(bic) else np.nan,
                'Converged': result['converged'],
                'MaxAbsA': max_abs_A,
            })

    results_df = pd.DataFrame(rows)

    # Save table
    results_df.to_csv(RESULTS_AGG_DIR / 'empirical_fit_table.csv', index=False)
    print(results_df.to_string())

    # Create regime classification figures
    yields2 = load_treasury_yields(MATURITIES)
    for mat in MATURITIES:
        key = (mat, 'Constant')
        if key in all_results and all_results[key]['converged']:
            y_diff = y_diff_for_mat(yields2, mat)
            dates = yields2.index[1:]             # after differencing
            _plot_regime_classification(
                dates, y_diff,
                all_results[key]['filtered_probs'],
                all_results[key]['params'],
                maturity=mat,
            )

    return results_df


# ---------------------------------------------------------------------------
# Regime classification figure
# ---------------------------------------------------------------------------

def _plot_regime_classification(
    dates: pd.DatetimeIndex,
    y: np.ndarray,
    filtered_probs: np.ndarray,
    params: Dict,
    maturity: int,
) -> None:
    """
    Plot yield changes with regime shading.
    Regimes ordered by increasing variance: low=blue, medium=salmon, high=red.
    """
    if filtered_probs is None or len(filtered_probs) == 0:
        return

    # Ensure dates and y align to filtered_probs length
    n = len(filtered_probs)
    if len(dates) > n:
        dates = dates[:n]
    if len(y) > n:
        y = y[:n]

    regime_order = np.argsort(params['sigma2'])   # low → high variance
    colors = ['#4472C4', '#FFA07A', '#C00000']     # blue, salmon, red

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y, 'k-', linewidth=0.5, alpha=0.7)

    most_prob_regime = np.argmax(filtered_probs[:n], axis=1)

    for t in range(n):
        regime_idx = int(np.where(regime_order == most_prob_regime[t])[0][0])
        t_next = min(t + 1, n - 1)
        ax.axvspan(dates[t], dates[t_next],
                   color=colors[regime_idx], alpha=0.25, linewidth=0)

    ax.set_title(f'Treasury Yield Changes ({maturity}m) — K=3 Regime Classification')
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield Change (bps)')
    fig.tight_layout()
    out_path = FIGURES_DIR / f'regime_classification_{maturity}m.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Figure saved: {out_path}')


if __name__ == '__main__':
    estimate_all_maturities()
