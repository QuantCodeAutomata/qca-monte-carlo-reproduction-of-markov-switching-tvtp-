"""
Performance metrics for Monte Carlo simulation study.

Computes:
  - Bias, RMSE, and 95% Wald coverage for each parameter
  - MAFE and MSFE for one-step-ahead regime-mean forecasts
  - MSE and MAE of filtered probability errors vs. reference (true-parameter) filter
"""

import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Parameter-level metrics (bias, RMSE, coverage)
# ---------------------------------------------------------------------------

def compute_parameter_metrics(
    estimates: List[np.ndarray],
    true_params: np.ndarray,
    ses: List[np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Compute bias, RMSE, and 95% Wald CI coverage across Monte Carlo replications.

    Parameters
    ----------
    estimates  : list of R arrays, each of length P (estimated parameter vectors)
    true_params: (P,) array of true parameter values
    ses        : list of R arrays, each of length P (standard error estimates)
    alpha      : significance level for two-sided Wald CI (default 0.05 → 95% CI)

    Returns
    -------
    dict with keys:
      'bias'     : (P,) mean of (estimate - true) across replications
      'rmse'     : (P,) root-mean-square error
      'coverage' : (P,) fraction of 95% Wald CIs that contain the true value
      'mean_se'  : (P,) mean standard error across replications
    """
    true_params = np.asarray(true_params, dtype=float)
    P = len(true_params)
    R = len(estimates)

    est_matrix = np.full((R, P), np.nan)
    se_matrix = np.full((R, P), np.nan)

    for r, (est, se) in enumerate(zip(estimates, ses)):
        est = np.asarray(est, dtype=float)
        se = np.asarray(se, dtype=float)
        n = min(len(est), P)
        est_matrix[r, :n] = est[:n]
        se_matrix[r, :n] = se[:n]

    errors = est_matrix - true_params[np.newaxis, :]  # (R, P)

    bias = np.nanmean(errors, axis=0)
    rmse = np.sqrt(np.nanmean(errors ** 2, axis=0))
    mean_se = np.nanmean(se_matrix, axis=0)

    # 95% Wald CI: [est - z * se, est + z * se]
    z = 1.96  # approx. quantile for alpha=0.05
    lower = est_matrix - z * se_matrix
    upper = est_matrix + z * se_matrix
    in_ci = (lower <= true_params[np.newaxis, :]) & (true_params[np.newaxis, :] <= upper)
    coverage = np.nanmean(in_ci.astype(float), axis=0)

    return {
        'bias': bias,
        'rmse': rmse,
        'coverage': coverage,
        'mean_se': mean_se,
    }


# ---------------------------------------------------------------------------
# Forecast accuracy metrics
# ---------------------------------------------------------------------------

def compute_forecast_metrics(
    y: np.ndarray,
    pred_probs: np.ndarray,
    mu_hat: np.ndarray,
    burn_in: int,
    end_trim: int,
) -> Dict[str, float]:
    """
    One-step-ahead point forecast accuracy over the estimation window.

    Forecast: yhat_{t|t-1} = sum_k mu_hat_k * pred_prob_{t,k}

    Parameters
    ----------
    y          : (T,) observed series
    pred_probs : (T, K) one-step predicted regime probabilities from the filter
    mu_hat     : (K,) estimated regime means
    burn_in    : start of evaluation window (inclusive)
    end_trim   : number of observations trimmed from the end

    Returns
    -------
    dict with 'MAFE' and 'MSFE' computed over t in [burn_in, T-end_trim)
    """
    T = len(y)
    t_start = burn_in
    t_end = T - end_trim if end_trim > 0 else T

    y_win = y[t_start:t_end]
    pp_win = pred_probs[t_start:t_end]          # (n_win, K)

    yhat = pp_win @ np.asarray(mu_hat, dtype=float)  # (n_win,)
    errors = y_win - yhat

    return {
        'MAFE': float(np.mean(np.abs(errors))),
        'MSFE': float(np.mean(errors ** 2)),
    }


# ---------------------------------------------------------------------------
# Filtered probability errors
# ---------------------------------------------------------------------------

def compute_filtered_prob_errors(
    filtered_hat: np.ndarray,
    filtered_true: np.ndarray,
    burn_in: int,
    end_trim: int,
) -> Dict[str, float]:
    """
    Pointwise filtered probability errors vs. the reference filter run with
    true parameters.

    Parameters
    ----------
    filtered_hat  : (T, K) estimated filtered regime probabilities
    filtered_true : (T, K) reference filtered probs (from true parameters)
    burn_in       : start of evaluation window (inclusive)
    end_trim      : number of observations trimmed from the end

    Returns
    -------
    dict with 'MSE' and 'MAE' averaged over the evaluation window and all K
    columns.

    Notes
    -----
    The reference is the filter run with *true* DGP parameters, NOT the
    simulated latent state sequence.  This avoids label-switching issues
    when comparing probability paths.
    """
    T = filtered_hat.shape[0]
    t_start = burn_in
    t_end = T - end_trim if end_trim > 0 else T

    diff = filtered_hat[t_start:t_end] - filtered_true[t_start:t_end]

    return {
        'MSE': float(np.mean(diff ** 2)),
        'MAE': float(np.mean(np.abs(diff))),
    }


# ---------------------------------------------------------------------------
# Convergence and regime statistics
# ---------------------------------------------------------------------------

def compute_regime_duration(z: np.ndarray, K: int) -> Dict[str, np.ndarray]:
    """
    Compute average regime duration (number of consecutive periods in a regime).

    Parameters
    ----------
    z : (T,) integer array of latent states (0-indexed)
    K : number of regimes

    Returns
    -------
    dict with 'mean_duration' (length K) array of average sojourn times
    """
    durations = [[] for _ in range(K)]
    t = 0
    T = len(z)
    while t < T:
        k = z[t]
        start = t
        while t < T and z[t] == k:
            t += 1
        durations[k].append(t - start)

    mean_dur = np.array([
        float(np.mean(d)) if d else 0.0 for d in durations
    ])
    return {'mean_duration': mean_dur}


def compute_regime_frequency(z: np.ndarray, K: int) -> np.ndarray:
    """
    Compute unconditional regime frequencies from simulated latent states.

    Returns
    -------
    (K,) array of regime proportions
    """
    counts = np.bincount(z, minlength=K)
    return counts / len(z)
