"""
Experiment 1: Monte Carlo Reproduction of Markov-Switching TVTP Parameter Recovery.
Covering DGPs 1-9, K=2/K=3, T=500/1000, R=50 replications.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    R, N_STARTS, T_LIST, BURN_IN, END_TRIM, EFF_OBS,
    EXTREME_A_THRESHOLD, SEED_BASE,
)
from config.dgp_params import get_all_dgps
from src.simulate import simulate_dgp
from src.filter import hamilton_filter
from src.estimate import (
    estimate_model, correct_label_switching,
    pack_params,
)
from src.metrics import (
    compute_parameter_metrics,
    compute_forecast_metrics,
    compute_filtered_prob_errors,
)
from src.reporting import aggregate_mc_results, format_results_table

RESULTS_RAW_DIR = Path(__file__).parent.parent / 'results' / 'raw'
RESULTS_AGG_DIR = Path(__file__).parent.parent / 'results' / 'aggregated'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_true_omega(dgp) -> np.ndarray:
    """Return baseline logit parameters from the DGP (logit space)."""
    return dgp.get_omega()


def _build_true_theta(dgp, model: str) -> np.ndarray:
    """
    Build the true parameter vector (same layout as pack_params) for a
    given DGP and the *estimated* model.  Parameters absent from the
    estimated model are omitted; parameters absent from the DGP (e.g. A
    when DGP is Constant but model is ModelI) are set to zero.
    """
    omega = dgp.get_omega()
    n_t = len(omega)
    log_sigma2 = np.log(np.array(dgp.sigma2, dtype=float))

    true_A: Optional[np.ndarray] = None
    true_B: Optional[np.ndarray] = None

    if model in ('ModelI', 'ModelII', 'ModelIII'):
        true_A = np.array(dgp.A, dtype=float) if dgp.A is not None else np.zeros(n_t)
    if model == 'ModelIII':
        true_B = np.array(dgp.B, dtype=float) if dgp.B is not None else np.zeros(n_t)

    return pack_params(
        np.array(dgp.mu, dtype=float), log_sigma2, omega,
        A=true_A, B=true_B, model=model,
    )


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def run_single_replication(
    rep_id: int,
    dgp,
    T: int,
    model: str,
    seed: int,
) -> Dict:
    """
    Run a single Monte Carlo replication for a given DGP, T, and estimation model.
    Returns dict with all estimated parameters, metrics, flags.
    """
    rng = np.random.default_rng(seed)

    # 1. Simulate data
    y, z_true, X = simulate_dgp(dgp, T, rng)

    # 2. Get reference filtered probs using TRUE parameters
    true_omega = get_true_omega(dgp)
    true_A = np.array(dgp.A, dtype=float) if dgp.A is not None else None
    true_B = np.array(dgp.B, dtype=float) if dgp.B is not None else None
    true_sigma2 = np.array(
        dgp.sigma2 * dgp.K if dgp.common_variance else dgp.sigma2,
        dtype=float,
    )

    _, filtered_true, pred_true = hamilton_filter(
        y, np.array(dgp.mu), true_sigma2,
        dgp.K, dgp.parameterization, dgp.transition_model,
        true_omega, true_A, true_B, X,
        burn_in=BURN_IN, end_trim=END_TRIM,
    )

    # 3. Estimate the specified model
    X_input = X if model == 'ModelII' else None
    result = estimate_model(
        y, dgp.K, dgp.parameterization, model,
        dgp.common_variance, N_STARTS, BURN_IN, END_TRIM,
        rng, X=X_input,
    )

    if not result['converged']:
        return {
            'converged': False,
            'rep_id': rep_id,
            'dgp_id': dgp.dgp_id,
            'model': model,
            'T': T,
        }

    # 4. Label switching correction
    params, filtered_hat, pred_hat = correct_label_switching(
        result['params']['mu'], np.array(dgp.mu),
        result['params'], result['filtered_probs'], result['pred_probs'],
        dgp.K, dgp.parameterization,
    )

    # 5. Compute metrics
    forecast_metrics = compute_forecast_metrics(
        y, pred_hat, params['mu'], BURN_IN, END_TRIM,
    )
    filtered_metrics = compute_filtered_prob_errors(
        filtered_hat, filtered_true, BURN_IN, END_TRIM,
    )

    return {
        'converged': True,
        'rep_id': rep_id,
        'dgp_id': dgp.dgp_id,
        'model': model,
        'T': T,
        'params': params,
        'theta_hat': result['theta_hat'],
        'se': result['se'],
        'loglik': result['loglik'],
        'filtered_probs': filtered_hat,
        'pred_probs': pred_hat,
        'forecast_metrics': forecast_metrics,
        'filtered_metrics': filtered_metrics,
    }


# ---------------------------------------------------------------------------
# Master experiment runner
# ---------------------------------------------------------------------------

def run_mc_experiment(
    dgp_ids: Optional[List[int]] = None,
    T_list: Optional[List[int]] = None,
    n_jobs: int = -1,
) -> None:
    """
    Run full Monte Carlo experiment for selected DGPs and sample sizes.
    Saves raw results to results/raw/ and aggregated to results/aggregated/.
    """
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AGG_DIR.mkdir(parents=True, exist_ok=True)

    dgps = get_all_dgps()
    if dgp_ids:
        dgps = {k: v for k, v in dgps.items() if k in dgp_ids}
    T_list = T_list or T_LIST

    all_summary: Dict = {}

    for dgp_id, dgp in dgps.items():
        for T in T_list:
            for model in dgp.estimation_models:
                print(f'Running DGP {dgp_id}, T={T}, Model={model}...')

                # Generate seeds for R replications
                seeds = [
                    SEED_BASE + dgp_id * 10000 + T * 100 + r
                    for r in range(R)
                ]

                # Run replications (parallel)
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_replication)(r, dgp, T, model, seeds[r])
                    for r in range(R)
                )

                # Filter converged results
                converged = [res for res in results if res.get('converged', False)]
                n_converged = len(converged)
                print(f'  Converged: {n_converged}/{R}')

                # DGP7, T=500, ModelI: exclude replications with extreme |A|
                if dgp_id == 7 and T == 500 and model == 'ModelI':
                    valid = [
                        r for r in converged
                        if np.max(np.abs(r['params'].get('A', np.zeros(1)))) <= EXTREME_A_THRESHOLD
                    ]
                    n_excluded = n_converged - len(valid)
                    print(f'  DGP7 T=500 ModelI: excluded {n_excluded} extreme-A replications')
                    converged_for_agg = valid
                else:
                    converged_for_agg = converged

                # Save raw results
                raw_key = f'dgp{dgp_id}_T{T}_{model}'
                raw_path = RESULTS_RAW_DIR / f'{raw_key}.pkl'
                with open(raw_path, 'wb') as fh:
                    pickle.dump(results, fh)

                # Build true_theta for reporting
                true_theta = _build_true_theta(dgp, model)

                # Aggregate metrics
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    summary = aggregate_mc_results(
                        converged_for_agg, dgp, model, T, true_theta,
                    )
                all_summary[(dgp_id, model, T)] = summary

                # Save aggregated CSV
                if not summary.empty:
                    agg_path = RESULTS_AGG_DIR / f'{raw_key}.csv'
                    summary.to_csv(agg_path, index=False)

    # Save master summary and generate RESULTS.md
    _save_master_results(all_summary)


def _save_master_results(all_summary: Dict) -> None:
    """Concatenate all per-DGP summaries, save master CSV, write RESULTS.md."""
    frames = [df for df in all_summary.values() if df is not None and not df.empty]
    if not frames:
        print('No results to save.')
        return

    master = pd.concat(frames, ignore_index=True)
    master_csv = RESULTS_AGG_DIR / 'master_results.csv'
    master.to_csv(master_csv, index=False)
    print(f'Master results saved to {master_csv}')

    # Generate RESULTS.md
    try:
        md_text = format_results_table(all_summary)
    except Exception as exc:
        md_text = f'*Could not render table: {exc}*\n\nSee `{master_csv}` for raw data.'

    results_md = Path(__file__).parent.parent / 'RESULTS.md'
    with open(results_md, 'w') as fh:
        fh.write('# Monte Carlo Results\n\n')
        fh.write(md_text)
    print(f'RESULTS.md written to {results_md}')


if __name__ == '__main__':
    run_mc_experiment(n_jobs=1)  # use 1 job for reproducibility in testing
