"""
Experiment 2: GAS Identifiability Diagnostic.
Demonstrates non-identifiability of score coefficient A in Model III.
Uses DGP 4 (K=2) and DGP 9 (K=3) at T=500 and T=1000.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.dgp_params import get_all_dgps
from config.settings import BURN_IN, END_TRIM, SEED_BASE, N_STARTS
from src.filter import hamilton_filter
from src.simulate import simulate_dgp
from src.estimate import (
    estimate_model, pack_params, unpack_params, neg_loglik,
    correct_label_switching,
)
from src.metrics import compute_parameter_metrics

FIGURES_DIR = Path(__file__).parent.parent / 'results' / 'figures'
RESULTS_AGG_DIR = Path(__file__).parent.parent / 'results' / 'aggregated'

_GAS_DGPS = [4, 9]   # K=2 and K=3 GAS DGPs
_T_LIST = [500, 1000]


# ---------------------------------------------------------------------------
# 1. Collapse verification
# ---------------------------------------------------------------------------

def verify_gas_constant_collapse(
    y: np.ndarray,
    dgp,
    T: int,
) -> Dict[str, float]:
    """
    Verify that Model III with A=0 gives same log-likelihood and filtered
    probs as Constant model (when initialized from same omega parameters).

    Returns dict: {ll_constant, ll_gas_azero, ll_diff, passes}
    """
    # Estimate Constant model to obtain omega_hat (and other params)
    rng = np.random.default_rng(SEED_BASE)
    res_const = estimate_model(
        y, dgp.K, dgp.parameterization, 'Constant',
        dgp.common_variance, n_starts=N_STARTS,
        burn_in=BURN_IN, end_trim=END_TRIM, rng=rng,
    )

    if not res_const['converged']:
        return {'ll_constant': np.nan, 'll_gas_azero': np.nan,
                'll_diff': np.nan, 'passes': False}

    p_const = res_const['params']
    omega_hat = p_const['omega']
    n_t = len(omega_hat)

    # Evaluate ModelIII at A=0, B=arbitrary (e.g. 0.9) and same omega
    A_zero = np.zeros(n_t)
    B_any = np.full(n_t, 0.9)
    sigma2_const = p_const['sigma2']

    ll_const = res_const['loglik']

    try:
        ll_gas, _, _ = hamilton_filter(
            y, p_const['mu'], sigma2_const,
            dgp.K, dgp.parameterization, 'ModelIII',
            omega_hat, A=A_zero, B=B_any,
            burn_in=BURN_IN, end_trim=END_TRIM,
        )
    except Exception:
        ll_gas = np.nan

    ll_diff = abs(ll_const - ll_gas) if np.isfinite(ll_gas) else np.inf
    passes = ll_diff < 1e-6

    return {
        'll_constant': ll_const,
        'll_gas_azero': ll_gas,
        'll_diff': ll_diff,
        'passes': passes,
    }


# ---------------------------------------------------------------------------
# 2. A-RMSE does not improve with T
# ---------------------------------------------------------------------------

def check_a_rmse_vs_T(
    n_rep: int = 20,
) -> pd.DataFrame:
    """
    Run n_rep replications at T=500 and T=1000 for DGPs 4 and 9.
    Compute A-RMSE for ModelIII.  Non-identifiability → RMSE does not
    shrink with T.

    Returns DataFrame with columns: dgp_id, T, a_rmse, n_rep
    """
    dgps = get_all_dgps()
    rows = []

    for dgp_id in _GAS_DGPS:
        dgp = dgps[dgp_id]
        true_A = np.array(dgp.A, dtype=float) if dgp.A is not None else None
        if true_A is None:
            continue

        for T in _T_LIST:
            A_estimates = []

            for r in range(n_rep):
                seed = SEED_BASE + dgp_id * 10000 + T * 100 + r
                rng = np.random.default_rng(seed)
                y, _, _ = simulate_dgp(dgp, T, rng)

                rng2 = np.random.default_rng(seed + 1)
                res = estimate_model(
                    y, dgp.K, dgp.parameterization, 'ModelIII',
                    dgp.common_variance, n_starts=3,
                    burn_in=BURN_IN, end_trim=END_TRIM, rng=rng2,
                )
                if res['converged'] and res['params'].get('A') is not None:
                    A_estimates.append(res['params']['A'])

            if A_estimates:
                A_mat = np.array(A_estimates)      # (n_conv, n_t)
                rmse = np.sqrt(np.mean((A_mat - true_A) ** 2))
            else:
                rmse = np.nan

            rows.append({
                'dgp_id': dgp_id, 'T': T, 'a_rmse': rmse,
                'n_rep': len(A_estimates),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Profile likelihood in A
# ---------------------------------------------------------------------------

def profile_likelihood_A(
    y: np.ndarray,
    dgp,
    base_params: Dict,
    A_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Profile log-likelihood over A_grid, holding other params at base_params.

    For K=2 diagonal, scans a single A value applied uniformly to both
    logit parameters.

    Parameters
    ----------
    y           : observed series
    dgp         : DGP configuration
    base_params : dict with mu, sigma2, omega (and B if ModelIII)
    A_grid      : 1D array of A values to scan

    Returns
    -------
    (A_grid, profile_ll) arrays for plotting
    """
    n_t = len(base_params['omega'])
    profile_ll = np.empty(len(A_grid))
    B = base_params.get('B', np.full(n_t, 0.9))

    for i, a_val in enumerate(A_grid):
        A = np.full(n_t, a_val)
        try:
            ll, _, _ = hamilton_filter(
                y, base_params['mu'], base_params['sigma2'],
                dgp.K, dgp.parameterization, 'ModelIII',
                base_params['omega'], A=A, B=B,
                burn_in=BURN_IN, end_trim=END_TRIM,
            )
            profile_ll[i] = ll if np.isfinite(ll) else np.nan
        except Exception:
            profile_ll[i] = np.nan

    return A_grid, profile_ll


# ---------------------------------------------------------------------------
# 4. 2D profile over (A, sigma^2)
# ---------------------------------------------------------------------------

def profile_likelihood_A_sigma2(
    y: np.ndarray,
    dgp,
    base_params: Dict,
    A_range: Tuple[float, float],
    sigma2_range: Tuple[float, float],
    n_grid: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D profile log-likelihood over A and sigma^2 (K=2 common variance only).

    Returns (A_vals, sigma2_vals, ll_surface) for contour plotting.
    """
    A_vals = np.linspace(A_range[0], A_range[1], n_grid)
    s2_vals = np.linspace(sigma2_range[0], sigma2_range[1], n_grid)
    ll_surface = np.full((n_grid, n_grid), np.nan)

    n_t = len(base_params['omega'])
    B = base_params.get('B', np.full(n_t, 0.9))

    for i, a_val in enumerate(A_vals):
        for j, s2 in enumerate(s2_vals):
            A = np.full(n_t, a_val)
            sigma2 = np.full(dgp.K, s2)
            try:
                ll, _, _ = hamilton_filter(
                    y, base_params['mu'], sigma2,
                    dgp.K, dgp.parameterization, 'ModelIII',
                    base_params['omega'], A=A, B=B,
                    burn_in=BURN_IN, end_trim=END_TRIM,
                )
                ll_surface[i, j] = ll if np.isfinite(ll) else np.nan
            except Exception:
                ll_surface[i, j] = np.nan

    return A_vals, s2_vals, ll_surface


# ---------------------------------------------------------------------------
# 5. Hessian diagnostics
# ---------------------------------------------------------------------------

def diagnose_hessian(
    result: Dict,
    dgp,
) -> Dict:
    """
    Check Hessian eigenvalues and condition number.
    Identify which dimensions (A parameters) have near-singular Hessian.

    Returns dict: eigenvalues, condition_number, near_singular_dims, n_neg_eigs
    """
    H = result.get('hessian')
    if H is None or not np.all(np.isfinite(H)):
        return {
            'eigenvalues': None,
            'condition_number': np.inf,
            'near_singular_dims': [],
            'n_neg_eigs': 0,
            'note': 'Hessian unavailable or non-finite',
        }

    try:
        eigvals = np.linalg.eigvalsh(H)
        pos_eigvals = eigvals[eigvals > 0]
        cond = float(eigvals.max() / eigvals.min()) if eigvals.min() > 0 else np.inf
        n_neg = int(np.sum(eigvals <= 0))

        # Near-singular: eigenvalue < 1e-6 * max
        threshold = 1e-6 * eigvals.max()
        near_singular_dims = [int(i) for i, v in enumerate(eigvals) if abs(v) < threshold]

        return {
            'eigenvalues': eigvals,
            'condition_number': cond,
            'near_singular_dims': near_singular_dims,
            'n_neg_eigs': n_neg,
        }
    except np.linalg.LinAlgError as exc:
        return {
            'eigenvalues': None,
            'condition_number': np.inf,
            'near_singular_dims': [],
            'n_neg_eigs': 0,
            'note': str(exc),
        }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_profile_A(A_grid: np.ndarray, profile_ll: np.ndarray,
                    dgp_id: int, T: int) -> None:
    """Figure 1: Profile LL for A (DGP 4 or 9, one dataset, T=500)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(A_grid, profile_ll, 'b-', linewidth=1.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1, label='A=0')
    ax.set_xlabel('Score coefficient A')
    ax.set_ylabel('Profile log-likelihood')
    ax.set_title(f'DGP {dgp_id} (T={T}): Profile LL over A')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'gas_profile_A_dgp{dgp_id}_T{T}.png', dpi=150)
    plt.close(fig)


def _plot_contour_A_sigma2(A_vals: np.ndarray, s2_vals: np.ndarray,
                            ll_surface: np.ndarray,
                            dgp_id: int, T: int) -> None:
    """Figure 2: 2D contour of (A, sigma^2) showing the ridge."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = np.isfinite(ll_surface)
    if valid.any():
        vmin = np.nanpercentile(ll_surface, 5)
        vmax = np.nanmax(ll_surface)
        cf = ax.contourf(A_vals, s2_vals, ll_surface.T,
                         levels=15, vmin=vmin, vmax=vmax, cmap='viridis')
        fig.colorbar(cf, ax=ax, label='Log-likelihood')
    ax.set_xlabel('Score coefficient A')
    ax.set_ylabel('sigma^2')
    ax.set_title(f'DGP {dgp_id} (T={T}): 2D Profile LL (A, sigma²)')
    ax.axvline(0, color='w', linestyle='--', linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'gas_contour_A_sigma2_dgp{dgp_id}_T{T}.png', dpi=150)
    plt.close(fig)


def _plot_a_distribution(A_estimates: np.ndarray, true_A: float,
                          dgp_id: int, T: int) -> None:
    """Figure 3: A estimate distribution across replications."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(A_estimates, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(true_A, color='red', linestyle='--', linewidth=1.5, label=f'True A={true_A:.2f}')
    ax.axvline(0, color='green', linestyle=':', linewidth=1.5, label='A=0')
    ax.set_xlabel('Estimated A')
    ax.set_ylabel('Frequency')
    ax.set_title(f'DGP {dgp_id} (T={T}): A estimate distribution (R replications)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'gas_a_distribution_dgp{dgp_id}_T{T}.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_gas_diagnostic(save_figures: bool = True) -> None:
    """
    Run all GAS diagnostic procedures and save results.

    1. Collapse verification for DGPs 4 and 9
    2. A-RMSE vs T comparison
    3. Likelihood profiles
    4. Hessian diagnostics
    5. Save diagnostic report
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AGG_DIR.mkdir(parents=True, exist_ok=True)

    dgps = get_all_dgps()
    report_rows = []

    # ------------------------------------------------------------------
    # 1. Collapse verification
    # ------------------------------------------------------------------
    print('=== 1. Collapse verification ===')
    for dgp_id in _GAS_DGPS:
        dgp = dgps[dgp_id]
        for T in _T_LIST:
            rng = np.random.default_rng(SEED_BASE + dgp_id)
            y, _, _ = simulate_dgp(dgp, T, rng)
            col = verify_gas_constant_collapse(y, dgp, T)
            print(f'  DGP{dgp_id} T={T}: ll_diff={col["ll_diff"]:.2e}  passes={col["passes"]}')
            report_rows.append({
                'check': 'collapse',
                'dgp_id': dgp_id, 'T': T,
                **{k: v for k, v in col.items() if k != 'passes'},
                'passes': col['passes'],
            })

    # ------------------------------------------------------------------
    # 2. A-RMSE vs T
    # ------------------------------------------------------------------
    print('=== 2. A-RMSE vs T ===')
    rmse_df = check_a_rmse_vs_T(n_rep=20)
    print(rmse_df.to_string(index=False))
    rmse_df.to_csv(RESULTS_AGG_DIR / 'gas_a_rmse_vs_T.csv', index=False)

    # Collect A estimates for distribution plot
    all_A_estimates: Dict = {}
    for dgp_id in _GAS_DGPS:
        dgp = dgps[dgp_id]
        for T in _T_LIST:
            A_ests = []
            for r in range(20):
                seed = SEED_BASE + dgp_id * 10000 + T * 100 + r
                rng = np.random.default_rng(seed)
                y, _, _ = simulate_dgp(dgp, T, rng)
                rng2 = np.random.default_rng(seed + 1)
                res = estimate_model(
                    y, dgp.K, dgp.parameterization, 'ModelIII',
                    dgp.common_variance, n_starts=3,
                    burn_in=BURN_IN, end_trim=END_TRIM, rng=rng2,
                )
                if res['converged'] and res['params'].get('A') is not None:
                    A_ests.append(res['params']['A'])
            all_A_estimates[(dgp_id, T)] = A_ests

    # ------------------------------------------------------------------
    # 3. Likelihood profiles (DGP 4, one dataset, T=500)
    # ------------------------------------------------------------------
    print('=== 3. Likelihood profiles ===')
    for dgp_id in _GAS_DGPS:
        dgp = dgps[dgp_id]
        T = 500
        rng = np.random.default_rng(SEED_BASE + dgp_id + 99)
        y, _, _ = simulate_dgp(dgp, T, rng)

        # Estimate Constant to get base_params for profiling
        rng2 = np.random.default_rng(SEED_BASE + 1)
        res_c = estimate_model(
            y, dgp.K, dgp.parameterization, 'Constant',
            dgp.common_variance, n_starts=N_STARTS,
            burn_in=BURN_IN, end_trim=END_TRIM, rng=rng2,
        )
        if not res_c['converged']:
            print(f'  DGP{dgp_id}: Constant model did not converge; skipping profiles.')
            continue

        base_params = dict(res_c['params'])
        base_params['B'] = np.full(len(base_params['omega']), 0.9)

        A_grid = np.linspace(-2.0, 2.0, 80)
        A_vals, profile_ll = profile_likelihood_A(y, dgp, base_params, A_grid)

        if save_figures:
            _plot_profile_A(A_vals, profile_ll, dgp_id, T)

        # 2D profile (only for K=2 for speed)
        if dgp.K == 2:
            s2_range = (
                max(0.05, float(base_params['sigma2'][0]) * 0.5),
                float(base_params['sigma2'][0]) * 2.0,
            )
            A_v, s2_v, ll_surf = profile_likelihood_A_sigma2(
                y, dgp, base_params,
                A_range=(-1.5, 1.5),
                sigma2_range=s2_range,
                n_grid=15,
            )
            if save_figures:
                _plot_contour_A_sigma2(A_v, s2_v, ll_surf, dgp_id, T)

    # ------------------------------------------------------------------
    # 4. Hessian diagnostics
    # ------------------------------------------------------------------
    print('=== 4. Hessian diagnostics ===')
    hess_rows = []
    for dgp_id in _GAS_DGPS:
        dgp = dgps[dgp_id]
        T = 500
        rng = np.random.default_rng(SEED_BASE + dgp_id + 200)
        y, _, _ = simulate_dgp(dgp, T, rng)
        rng2 = np.random.default_rng(SEED_BASE + 3)
        res = estimate_model(
            y, dgp.K, dgp.parameterization, 'ModelIII',
            dgp.common_variance, n_starts=N_STARTS,
            burn_in=BURN_IN, end_trim=END_TRIM, rng=rng2,
        )
        hd = diagnose_hessian(res, dgp)
        cond = hd['condition_number']
        print(f'  DGP{dgp_id}: cond={cond:.2e}  n_neg_eigs={hd["n_neg_eigs"]}')
        hess_rows.append({'dgp_id': dgp_id, 'cond_number': cond,
                          'n_neg_eigs': hd['n_neg_eigs']})

    pd.DataFrame(hess_rows).to_csv(
        RESULTS_AGG_DIR / 'gas_hessian_diagnostics.csv', index=False)

    # ------------------------------------------------------------------
    # 5. A-estimate distribution plots
    # ------------------------------------------------------------------
    if save_figures:
        for dgp_id in _GAS_DGPS:
            dgp = dgps[dgp_id]
            true_A0 = float(np.array(dgp.A)[0]) if dgp.A else 0.0
            for T in _T_LIST:
                A_ests = all_A_estimates.get((dgp_id, T), [])
                if A_ests:
                    A_arr = np.array(A_ests)[:, 0]  # first A component
                    _plot_a_distribution(A_arr, true_A0, dgp_id, T)

    # ------------------------------------------------------------------
    # Save report CSV
    # ------------------------------------------------------------------
    pd.DataFrame(report_rows).to_csv(
        RESULTS_AGG_DIR / 'gas_collapse_verification.csv', index=False)
    print('GAS diagnostic complete.')


if __name__ == '__main__':
    run_gas_diagnostic()
