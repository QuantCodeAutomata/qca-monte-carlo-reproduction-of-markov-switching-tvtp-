"""
Maximum-likelihood estimation for Markov-switching models.

Provides:
  - pack_params / unpack_params : consistent parameter vector serialisation
  - neg_loglik               : objective function for optimisation
  - numerical_hessian        : finite-difference Hessian
  - compute_se               : standard errors from information matrix
  - correct_label_switching  : permutation alignment to true regime order
  - estimate_model           : multistart MLE with L-BFGS-B / Nelder-Mead
"""

import warnings
import numpy as np
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from scipy.optimize import minimize, approx_fprime

from src.filter import hamilton_filter, build_transition_matrix, logit


# ---------------------------------------------------------------------------
# Helper: number of transition parameters
# ---------------------------------------------------------------------------

def _n_trans(K: int, parameterization: str) -> int:
    """Return number of logit transition parameters."""
    if parameterization == 'diagonal':
        return K
    return K * (K - 1)


# ---------------------------------------------------------------------------
# Parameter pack / unpack
# ---------------------------------------------------------------------------

def pack_params(
    mu: np.ndarray,
    log_sigma2: np.ndarray,
    omega: np.ndarray,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    model: str = 'Constant',
) -> np.ndarray:
    """
    Pack all model parameters into a single flat vector.

    Layout:  [mu (K), log_sigma2 (1 or K), omega (n_trans),
              A (n_trans, if TVTP), B (n_trans, if GAS)]
    """
    parts = [np.asarray(mu, dtype=float),
             np.asarray(log_sigma2, dtype=float),
             np.asarray(omega, dtype=float)]
    if model != 'Constant':
        if A is None:
            raise ValueError(f"{model} requires A.")
        parts.append(np.asarray(A, dtype=float))
    if model == 'ModelIII':
        if B is None:
            raise ValueError("ModelIII requires B.")
        parts.append(np.asarray(B, dtype=float))
    return np.concatenate(parts)


def unpack_params(
    theta: np.ndarray,
    K: int,
    parameterization: str,
    model: str,
    common_variance: bool,
) -> Dict:
    """
    Unpack a flat parameter vector into named components.

    Returns
    -------
    dict with keys: 'mu', 'sigma2', 'log_sigma2', 'omega', and optionally 'A', 'B'
    """
    theta = np.asarray(theta, dtype=float)
    n_s = 1 if common_variance else K
    n_t = _n_trans(K, parameterization)

    idx = 0
    mu = theta[idx: idx + K];       idx += K
    log_s2 = theta[idx: idx + n_s]; idx += n_s
    omega = theta[idx: idx + n_t];  idx += n_t

    sigma2 = np.exp(log_s2)
    if common_variance:
        sigma2 = np.full(K, float(sigma2[0]))

    result: Dict = {
        'mu': mu,
        'sigma2': sigma2,
        'log_sigma2': log_s2,
        'omega': omega,
    }

    if model != 'Constant':
        A = theta[idx: idx + n_t]; idx += n_t
        result['A'] = A
    if model == 'ModelIII':
        B = theta[idx: idx + n_t]; idx += n_t
        result['B'] = B

    return result


# ---------------------------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------------------------

def neg_loglik(
    theta: np.ndarray,
    y: np.ndarray,
    K: int,
    parameterization: str,
    model: str,
    common_variance: bool,
    burn_in: int,
    end_trim: int,
    X: Optional[np.ndarray] = None,
) -> float:
    """Negative log-likelihood (objective for minimisation)."""
    try:
        params = unpack_params(theta, K, parameterization, model, common_variance)
        ll, _, _ = hamilton_filter(
            y=y,
            mu=params['mu'],
            sigma2=params['sigma2'],
            K=K,
            parameterization=parameterization,
            transition_model=model,
            omega=params['omega'],
            A=params.get('A'),
            B=params.get('B'),
            X=X,
            burn_in=burn_in,
            end_trim=end_trim,
        )
        return -ll if np.isfinite(ll) else 1e10
    except Exception:
        return 1e10


# ---------------------------------------------------------------------------
# Numerical Hessian
# ---------------------------------------------------------------------------

def numerical_hessian(
    f,
    theta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Symmetric finite-difference Hessian of scalar function f at theta.
    Uses forward differences on the gradient.
    """
    n = len(theta)
    H = np.zeros((n, n))
    grad0 = approx_fprime(theta, f, eps)
    for i in range(n):
        theta_i = theta.copy()
        theta_i[i] += eps
        grad_i = approx_fprime(theta_i, f, eps)
        H[i, :] = (grad_i - grad0) / eps
    return (H + H.T) / 2.0


# ---------------------------------------------------------------------------
# Standard errors
# ---------------------------------------------------------------------------

def compute_se(hessian: np.ndarray) -> np.ndarray:
    """
    Compute standard errors from the Hessian of the *negative* log-likelihood.
    Inverts the Hessian to obtain the asymptotic covariance matrix.
    Returns NaN for any SE that is non-positive or where inversion fails.
    """
    try:
        cov = np.linalg.inv(hessian)
        diag = np.diag(cov)
        return np.where(diag > 0, np.sqrt(diag), np.nan)
    except np.linalg.LinAlgError:
        return np.full(len(np.diag(hessian)), np.nan)


# ---------------------------------------------------------------------------
# Label switching correction
# ---------------------------------------------------------------------------

def _permute_trans_params(
    vec: np.ndarray,
    K: int,
    parameterization: str,
    perm: Tuple[int, ...],
) -> np.ndarray:
    """
    Reorder a transition parameter vector under regime permutation `perm`.

    For diagonal:  vec[k] corresponds to regime k's diagonal logit (f_kk).
    For off-diagonal: vec follows row-major off-diagonal ordering.
    """
    n_t = _n_trans(K, parameterization)
    new_vec = np.zeros(n_t)

    if parameterization == 'diagonal':
        # vec[k] = f_{p(k), p(k)} in new labelling -> comes from old regime perm[k]
        for k_new in range(K):
            k_old = perm[k_new]
            new_vec[k_new] = vec[k_old]

    else:
        # Build (row, col) -> index mapping for off-diagonal elements
        off_pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
        pair_to_idx = {p: idx for idx, p in enumerate(off_pairs)}
        for idx, (i_old, j_old) in enumerate(off_pairs):
            i_new = perm.index(i_old)
            j_new = perm.index(j_old)
            new_idx = pair_to_idx[(i_new, j_new)]
            new_vec[new_idx] = vec[idx]

    return new_vec


def correct_label_switching(
    estimated_mu: np.ndarray,
    true_mu: np.ndarray,
    all_estimated_params: Dict,
    filtered_probs: np.ndarray,
    pred_probs: np.ndarray,
    K: int,
    parameterization: str,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Find the regime permutation minimising sum_i |mu_hat_{perm(i)} - mu_true_i|
    and apply it to all parameters, filtered probabilities, and prediction probs.

    Returns
    -------
    corrected_params  : parameter dict with permuted ordering
    corrected_filtered: (T, K) filtered probs with permuted columns
    corrected_pred    : (T, K) predicted probs with permuted columns
    """
    best_perm = min(
        permutations(range(K)),
        key=lambda p: sum(
            abs(estimated_mu[p[i]] - true_mu[i]) for i in range(K)
        ),
    )
    perm = list(best_perm)

    corrected = dict(all_estimated_params)
    corrected['mu'] = estimated_mu[perm]

    sigma2 = all_estimated_params['sigma2']
    corrected['sigma2'] = sigma2[perm]

    for key in ('omega', 'A', 'B'):
        if key in all_estimated_params and all_estimated_params[key] is not None:
            corrected[key] = _permute_trans_params(
                all_estimated_params[key], K, parameterization, tuple(perm)
            )

    corrected_filtered = filtered_probs[:, perm]
    corrected_pred = pred_probs[:, perm]

    return corrected, corrected_filtered, corrected_pred


# ---------------------------------------------------------------------------
# Random start initialisation
# ---------------------------------------------------------------------------

def _random_start(
    K: int,
    parameterization: str,
    model: str,
    common_variance: bool,
    rng: np.random.Generator,
    true_params: Optional[Dict] = None,
) -> np.ndarray:
    """Draw one random starting point for the optimiser."""
    n_s = 1 if common_variance else K
    n_t = _n_trans(K, parameterization)

    if true_params is not None:
        mu0 = rng.normal(true_params['mu'], 0.5)
        log_s2_true = np.log(
            true_params['sigma2'][:n_s] if n_s > 1
            else true_params['sigma2'][:1]
        )
        log_s2 = rng.normal(log_s2_true, 0.3)
        omega0 = rng.normal(true_params['omega'], 0.5)
    else:
        # Spread means to avoid degenerate starts
        mu0 = rng.normal(0.0, 1.0, size=K)
        mu0 = np.sort(mu0)   # ordered start helps convergence
        log_s2 = rng.uniform(-2.0, 1.0, size=n_s)
        omega0 = rng.normal(0.0, 0.5, size=n_t)

    parts = [mu0, log_s2, omega0]

    if model != 'Constant':
        A0 = rng.normal(0.0, 0.3, size=n_t)
        parts.append(A0)
    if model == 'ModelIII':
        # Persistence near 0.8-0.9 is typical
        B0 = rng.normal(0.85, 0.1, size=n_t)
        B0 = np.clip(B0, -0.999, 0.999)
        parts.append(B0)

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_model(
    y: np.ndarray,
    K: int,
    parameterization: str,
    model: str,
    common_variance: bool,
    n_starts: int,
    burn_in: int,
    end_trim: int,
    rng: np.random.Generator,
    X: Optional[np.ndarray] = None,
    true_params: Optional[Dict] = None,
) -> Dict:
    """
    Estimate a Markov-switching model by multistart MLE.

    Tries `n_starts` random initialisations, retains the best converged
    result.  Primary optimiser is L-BFGS-B; falls back to Nelder-Mead when
    L-BFGS-B does not converge.

    Parameters
    ----------
    y              : observed series
    K              : number of regimes
    parameterization : 'diagonal' or 'off_diagonal'
    model          : 'Constant', 'ModelI', 'ModelII', 'ModelIII'
    common_variance: whether all regimes share one variance parameter
    n_starts       : number of random starting points
    burn_in        : observations excluded at start for log-likelihood
    end_trim       : observations excluded at end for log-likelihood
    rng            : NumPy Generator
    X              : exogenous driver (ModelII)
    true_params    : optional dict with true parameters for guided init

    Returns
    -------
    dict with keys:
      params         : estimated parameter dict (mu, sigma2, omega, A, B)
      loglik         : maximised log-likelihood
      theta_hat      : raw parameter vector at MLE
      converged      : whether at least one start converged
      hessian        : numerical Hessian of neg-loglik at MLE
      se             : standard errors (same length as theta_hat)
      n_converged    : number of converged starts
      filtered_probs : (T, K) filtered regime probabilities
      pred_probs     : (T, K) one-step predicted probabilities
    """
    obj = lambda theta: neg_loglik(
        theta, y, K, parameterization, model, common_variance,
        burn_in, end_trim, X
    )

    best_val = np.inf
    best_theta = None
    best_result = None
    n_converged = 0

    for _ in range(n_starts):
        theta0 = _random_start(K, parameterization, model, common_variance,
                               rng, true_params)
        # Primary: L-BFGS-B  (maxfun caps total filter evaluations;
        # with numba ≈0.5 ms/call, 1500 evals ≈ 0.75 s per start)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(obj, theta0, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 1500,
                                    'ftol': 1e-10, 'gtol': 1e-6})

        if not res.success or not np.isfinite(res.fun):
            # Fallback: Nelder-Mead
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(obj, theta0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-7,
                                        'fatol': 1e-9})

        if np.isfinite(res.fun):
            n_converged += 1
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
                best_result = res

    if best_theta is None:
        # All starts failed — return degenerate result
        n_t = _n_trans(K, parameterization)
        n_s = 1 if common_variance else K
        n_params = K + n_s + n_t
        best_theta = np.zeros(n_params)
        best_val = 1e10

    # Unpack best parameters
    params = unpack_params(best_theta, K, parameterization, model, common_variance)

    # Compute filtered probabilities at MLE
    ll, filtered_probs, pred_probs = hamilton_filter(
        y=y,
        mu=params['mu'],
        sigma2=params['sigma2'],
        K=K,
        parameterization=parameterization,
        transition_model=model,
        omega=params['omega'],
        A=params.get('A'),
        B=params.get('B'),
        X=X,
        burn_in=burn_in,
        end_trim=end_trim,
    )

    # Numerical Hessian for standard errors
    try:
        H = numerical_hessian(obj, best_theta)
        se = compute_se(H)
    except Exception:
        H = np.full((len(best_theta), len(best_theta)), np.nan)
        se = np.full(len(best_theta), np.nan)

    return {
        'params': params,
        'loglik': -best_val,
        'theta_hat': best_theta,
        'converged': n_converged > 0,
        'hessian': H,
        'se': se,
        'n_converged': n_converged,
        'filtered_probs': filtered_probs,
        'pred_probs': pred_probs,
    }
