"""
Monte Carlo results aggregation and report formatting.

Aggregates per-replication outputs into summary statistics and renders
them as Markdown or LaTeX tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config.dgp_params import DGPConfig
from src.metrics import compute_parameter_metrics, compute_forecast_metrics


# ---------------------------------------------------------------------------
# Aggregate MC results
# ---------------------------------------------------------------------------

def aggregate_mc_results(
    all_results: List[Dict],
    dgp: DGPConfig,
    model: str,
    T: int,
    true_theta: np.ndarray,
    param_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate R replication results into per-parameter summary statistics.

    Parameters
    ----------
    all_results : list of dicts returned by estimate_model (length R)
    dgp         : DGP configuration (used for true parameter values)
    model       : estimated transition model label
    T           : sample size used in this experiment
    true_theta  : (P,) true parameter vector (same ordering as theta_hat)
    param_names : optional list of P parameter name strings

    Returns
    -------
    pd.DataFrame with columns:
      Parameter, True Value, Bias, RMSE, Coverage95, Mean SE,
      DGP, Model, T, n_converged
    """
    R = len(all_results)
    estimates = []
    ses = []
    n_conv = 0

    for res in all_results:
        if res is None:
            continue
        theta = res.get('theta_hat')
        se = res.get('se')
        if theta is not None and se is not None:
            estimates.append(theta)
            ses.append(se)
        if res.get('converged', False):
            n_conv += 1

    if not estimates:
        return pd.DataFrame()

    metrics = compute_parameter_metrics(estimates, true_theta, ses)
    P = len(true_theta)

    if param_names is None:
        param_names = [f'param_{i}' for i in range(P)]

    rows = []
    for i, name in enumerate(param_names[:P]):
        rows.append({
            'Parameter':  name,
            'True Value': float(true_theta[i]),
            'Bias':       float(metrics['bias'][i]),
            'RMSE':       float(metrics['rmse'][i]),
            'Coverage95': float(metrics['coverage'][i]),
            'Mean SE':    float(metrics['mean_se'][i]),
            'DGP':        dgp.dgp_id,
            'Model':      model,
            'T':          T,
            'n_converged': n_conv,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate forecast metrics across replications
# ---------------------------------------------------------------------------

def aggregate_forecast_metrics(
    all_results: List[Dict],
    y_list: List[np.ndarray],
    burn_in: int,
    end_trim: int,
) -> Dict[str, float]:
    """
    Average MAFE and MSFE across replications.

    Parameters
    ----------
    all_results : list of estimate_model outputs
    y_list      : list of observed series (one per replication)
    burn_in     : burn-in period
    end_trim    : end trim

    Returns
    -------
    dict with 'mean_MAFE', 'mean_MSFE', 'std_MAFE', 'std_MSFE'
    """
    mafes, msfes = [], []
    for res, y in zip(all_results, y_list):
        if res is None:
            continue
        mu_hat = res['params'].get('mu')
        pred_probs = res.get('pred_probs')
        if mu_hat is None or pred_probs is None:
            continue
        fm = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in, end_trim)
        mafes.append(fm['MAFE'])
        msfes.append(fm['MSFE'])

    if not mafes:
        return {'mean_MAFE': np.nan, 'mean_MSFE': np.nan,
                'std_MAFE': np.nan, 'std_MSFE': np.nan}

    return {
        'mean_MAFE': float(np.mean(mafes)),
        'mean_MSFE': float(np.mean(msfes)),
        'std_MAFE':  float(np.std(mafes)),
        'std_MSFE':  float(np.std(msfes)),
    }


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def format_results_table(
    aggregated_results: Dict[Tuple, pd.DataFrame],
    metric_cols: Optional[List[str]] = None,
) -> str:
    """
    Concatenate aggregated results DataFrames into a Markdown table.

    Parameters
    ----------
    aggregated_results : dict keyed by (dgp_id, model, T), values are DataFrames
    metric_cols        : columns to include; defaults to all numeric metric columns

    Returns
    -------
    Markdown-formatted string
    """
    if not aggregated_results:
        return "No results to display."

    frames = []
    for (dgp_id, model, T), df in sorted(aggregated_results.items()):
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return "No results to display."

    combined = pd.concat(frames, ignore_index=True)

    if metric_cols is None:
        metric_cols = ['DGP', 'Model', 'T', 'Parameter', 'True Value',
                       'Bias', 'RMSE', 'Coverage95', 'Mean SE']

    available = [c for c in metric_cols if c in combined.columns]
    combined = combined[available]

    # Markdown formatting
    lines = ['| ' + ' | '.join(str(c) for c in available) + ' |']
    lines.append('|' + '---|' * len(available))

    for _, row in combined.iterrows():
        def fmt(v):
            if isinstance(v, float):
                return f'{v:.4f}'
            return str(v)
        lines.append('| ' + ' | '.join(fmt(row[c]) for c in available) + ' |')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# LaTeX table formatter
# ---------------------------------------------------------------------------

def format_latex_table(
    aggregated_results: Dict[Tuple, pd.DataFrame],
    caption: str = 'Monte Carlo Results',
    label: str = 'tab:mc_results',
) -> str:
    """
    Render aggregated results as a LaTeX longtable.

    Parameters
    ----------
    aggregated_results : dict keyed by (dgp_id, model, T)
    caption            : table caption
    label              : LaTeX label

    Returns
    -------
    LaTeX string
    """
    frames = []
    for (dgp_id, model, T), df in sorted(aggregated_results.items()):
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return ''

    combined = pd.concat(frames, ignore_index=True)
    cols = ['DGP', 'Model', 'T', 'Parameter', 'True Value',
            'Bias', 'RMSE', 'Coverage95']
    available = [c for c in cols if c in combined.columns]

    header = ' & '.join(available) + r' \\'
    hline = r'\hline'

    lines = [
        r'\begin{longtable}{' + 'l' * len(available) + '}',
        r'\caption{' + caption + r'} \label{' + label + r'} \\',
        hline, header, hline,
        r'\endfirsthead',
        hline, header, hline,
        r'\endhead',
        hline,
        r'\endfoot',
    ]

    def fmt_cell(v):
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)

    for _, row in combined.iterrows():
        lines.append(' & '.join(fmt_cell(row[c]) for c in available) + r' \\')

    lines += [r'\end{longtable}']
    return '\n'.join(lines)
