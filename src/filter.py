"""
Hamilton filter with time-varying transition probability (TVTP) support.

Implements all four transition specifications:
  - Constant   : fixed transition matrix
  - ModelI     : logit_t = omega + A * y_{t-1}
  - ModelII    : logit_t = omega + A * X_{t-1}
  - ModelIII   : GAS recursion with OPG-style Fisher scaling

Custom implementation — no mainstream Python library handles TVTP-GAS
Markov switching with off-diagonal row rescaling, exact paper likelihood
trimming, and the GAS score recursion described in Bazzi et al. (2017).
"""

import warnings
import numpy as np
from scipy.special import expit
from typing import Optional, Tuple

# Numba-JIT kernels (import may be skipped in minimal test environments)
try:
    from src.filter_numba import (
        nb_filter_K2_diag,
        nb_filter_K2_offdiag,
        nb_filter_K3_offdiag,
        warmup as _nb_warmup,
    )
    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False

# Map model name → integer code used by numba kernels
_MODEL_CODE = {'Constant': 0, 'ModelI': 1, 'ModelII': 2, 'ModelIII': 3}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic (sigmoid) function."""
    return expit(np.asarray(x, dtype=float))


def logit(p: np.ndarray) -> np.ndarray:
    """Element-wise logit, clipped for numerical safety."""
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1.0 - 1e-10)
    return np.log(p / (1.0 - p))


def gaussian_pdf(y: float, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Evaluate Gaussian densities N(y; mu_k, sigma2_k) for each regime k.
    Returns array of shape (K,), clipped below at 1e-300 for numerical safety.
    """
    diff = y - mu
    pdf = np.exp(-0.5 * diff ** 2 / sigma2) / np.sqrt(2.0 * np.pi * sigma2)
    return np.maximum(pdf, 1e-300)


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_distribution(Pi: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a row-stochastic transition matrix.
    Solves the linear system (Pi^T - I) pi = 0 subject to sum(pi) = 1.
    Falls back to uniform distribution if the system is singular.
    """
    K = Pi.shape[0]
    A = np.vstack([Pi.T - np.eye(K), np.ones((1, K))])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.abs(pi)
    total = pi.sum()
    if total < 1e-12 or not np.isfinite(total):
        return np.ones(K) / K
    return pi / total


# ---------------------------------------------------------------------------
# Transition matrix builder
# ---------------------------------------------------------------------------

def build_transition_matrix(
    f: np.ndarray,
    K: int,
    parameterization: str,
) -> np.ndarray:
    """
    Build a K×K row-stochastic transition matrix from logit-scale parameters.

    Parameters
    ----------
    f : logit-scale parameters
        K=2 diagonal      : [f_11, f_22]
        K=2 off_diagonal  : [f_12, f_21]
        K=3 off_diagonal  : [f_12, f_13, f_21, f_23, f_31, f_32]
    K : number of regimes
    parameterization : 'diagonal' or 'off_diagonal'

    Returns
    -------
    Pi : (K, K) row-stochastic matrix where Pi[i, j] = P(z_t=j | z_{t-1}=i)

    Notes
    -----
    For K=3 off_diagonal, when the sum of off-diagonal probabilities in a row
    exceeds 1, all off-diagonal entries are rescaled proportionally so that
    the row sums to 1.  This rescaling is critical for parameter identifiability
    in the off-diagonal specification.
    """
    f = np.asarray(f, dtype=float)
    Pi = np.zeros((K, K))

    if K == 2 and parameterization == 'diagonal':
        p11 = float(sigmoid(f[0]))
        Pi[0, 0] = p11;      Pi[0, 1] = 1.0 - p11
        p22 = float(sigmoid(f[1]))
        Pi[1, 1] = p22;      Pi[1, 0] = 1.0 - p22

    elif K == 2 and parameterization == 'off_diagonal':
        p12 = float(sigmoid(f[0]))
        Pi[0, 1] = p12;      Pi[0, 0] = 1.0 - p12
        p21 = float(sigmoid(f[1]))
        Pi[1, 0] = p21;      Pi[1, 1] = 1.0 - p21

    elif K == 3 and parameterization == 'off_diagonal':
        # f = [f_12, f_13, f_21, f_23, f_31, f_32]  (row-major off-diagonal order)
        for i in range(3):
            j_cols = [j for j in range(3) if j != i]   # two off-diagonal column indices
            f_row = f[2 * i: 2 * i + 2]                # logit params for this row
            p_off = sigmoid(f_row).copy()               # raw off-diagonal probs
            s = p_off.sum()
            if s > 1.0 - 1e-10:
                # Proportional rescaling so row sums to 1
                p_off = p_off / (s + 1e-10)
            Pi[i, j_cols] = p_off
            Pi[i, i] = max(0.0, 1.0 - p_off.sum())

    else:
        raise ValueError(
            f"Unsupported configuration: K={K}, parameterization='{parameterization}'"
        )

    return Pi


# ---------------------------------------------------------------------------
# GAS score computation (Model III)
# ---------------------------------------------------------------------------

def _gas_score(
    xi_prev: np.ndarray,
    phi_t: np.ndarray,
    Pi_t: np.ndarray,
    p_t: float,
    K: int,
    parameterization: str,
) -> np.ndarray:
    """
    Compute the unscaled score of log p_t with respect to the logit parameters f_t.

    Uses closed-form derivatives of the predictive log-likelihood w.r.t. each
    logit parameter.  For K=3 the rescale-factor correction is omitted (≈ 1
    when off-diagonal probabilities are small, as in all study DGPs).

    Custom GAS score — no library equivalent for exact Bazzi et al. (2017)
    scaling.  Using unscaled score (S=1) as baseline; Fisher info is handled
    separately via OPG EMA.
    """
    p_safe = max(p_t, 1e-300)

    if K == 2 and parameterization == 'diagonal':
        # f = [f_11, f_22];  pi_ii = sigmoid(f_ii)
        pi11 = Pi_t[0, 0];  sigma_prime_11 = pi11 * (1.0 - pi11)
        pi22 = Pi_t[1, 1];  sigma_prime_22 = pi22 * (1.0 - pi22)
        nabla = np.array([
            xi_prev[0] * sigma_prime_11 * (phi_t[0] - phi_t[1]) / p_safe,
            xi_prev[1] * sigma_prime_22 * (phi_t[1] - phi_t[0]) / p_safe,
        ])

    elif K == 2 and parameterization == 'off_diagonal':
        # f = [f_12, f_21];  pi_ij = sigmoid(f_ij)
        pi12 = Pi_t[0, 1];  sigma_prime_12 = pi12 * (1.0 - pi12)
        pi21 = Pi_t[1, 0];  sigma_prime_21 = pi21 * (1.0 - pi21)
        nabla = np.array([
            xi_prev[0] * sigma_prime_12 * (phi_t[1] - phi_t[0]) / p_safe,
            xi_prev[1] * sigma_prime_21 * (phi_t[0] - phi_t[1]) / p_safe,
        ])

    elif K == 3 and parameterization == 'off_diagonal':
        # f = [f_12,f_13, f_21,f_23, f_31,f_32]
        # nabla[2i]   = score for off-diag element f_{i,j1}
        # nabla[2i+1] = score for off-diag element f_{i,j2}
        nabla = np.zeros(6)
        j_pairs = [(1, 2), (0, 2), (0, 1)]  # off-diagonal cols for each row i
        for i in range(3):
            j1, j2 = j_pairs[i]
            pi_ij1 = Pi_t[i, j1];  sp1 = pi_ij1 * (1.0 - pi_ij1)
            pi_ij2 = Pi_t[i, j2];  sp2 = pi_ij2 * (1.0 - pi_ij2)
            nabla[2 * i]     = xi_prev[i] * sp1 * (phi_t[j1] - phi_t[i]) / p_safe
            nabla[2 * i + 1] = xi_prev[i] * sp2 * (phi_t[j2] - phi_t[i]) / p_safe

    else:
        raise ValueError(
            f"GAS score not implemented for K={K}, param='{parameterization}'"
        )

    return nabla


# ---------------------------------------------------------------------------
# Hamilton filter
# ---------------------------------------------------------------------------

def hamilton_filter(
    y: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    K: int,
    parameterization: str,
    transition_model: str,
    omega: np.ndarray,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    burn_in: int = 100,
    end_trim: int = 0,
    return_filtered: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run the Hamilton (1989) filter for a Markov-switching model with optional
    time-varying transition probabilities.

    Parameters
    ----------
    y            : observed series, shape (T,)
    mu           : regime means, shape (K,)
    sigma2       : regime variances, shape (K,)
    K            : number of regimes
    parameterization : 'diagonal' or 'off_diagonal'
    transition_model : 'Constant', 'ModelI', 'ModelII', 'ModelIII'
    omega        : baseline logit parameters
    A            : TVTP score coefficients (ModelI/II/III)
    B            : GAS persistence coefficients (ModelIII only)
    X            : exogenous driver for ModelII, shape (T,)
    burn_in      : first `burn_in` observations excluded from log-likelihood
    end_trim     : last `end_trim` observations excluded from log-likelihood
    return_filtered : (unused flag kept for API compatibility)

    Returns
    -------
    loglik        : scalar log-likelihood over [burn_in, T-end_trim)
    filtered_probs  : (T, K) filtered regime probabilities for all T periods
    pred_probs_all  : (T, K) one-step-ahead predicted regime probabilities

    Notes
    -----
    Initialization: xi_prev = stationary distribution of the baseline
    transition matrix (Pi evaluated at f=omega).  This avoids dependence on
    an arbitrary starting point.

    For ModelIII (GAS):
      f_{t+1} = omega + A * s_t + B * (f_t - omega)
    where s_t = nabla_t / sqrt(I_t + eps) is the Fisher-scaled score and
    I_t is an OPG EMA of squared scores with decay rho=0.95.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    omega = np.asarray(omega, dtype=float)
    T = len(y)
    n_f = len(omega)

    # ── Numba fast-path (≈50× faster than pure-Python loop) ─────────────
    if _NUMBA_OK and transition_model in _MODEL_CODE:
        mt   = _MODEL_CODE[transition_model]
        A_nb = np.asarray(A, dtype=float) if A is not None else np.zeros(n_f)
        B_nb = np.asarray(B, dtype=float) if B is not None else np.zeros(n_f)
        X_nb = np.asarray(X, dtype=float) if X is not None else np.zeros(T)
        et   = end_trim if end_trim > 0 else 0

        if K == 2 and parameterization == 'diagonal':
            return nb_filter_K2_diag(
                y, mu, sigma2, omega, A_nb, B_nb, X_nb, mt, burn_in, et)
        elif K == 2 and parameterization == 'off_diagonal':
            return nb_filter_K2_offdiag(
                y, mu, sigma2, omega, A_nb, B_nb, X_nb, mt, burn_in, et)
        elif K == 3 and parameterization == 'off_diagonal':
            return nb_filter_K3_offdiag(
                y, mu, sigma2, omega, A_nb, B_nb, X_nb, mt, burn_in, et)
        # else: fall through to pure-Python path below

    # ── Pure-Python fallback ─────────────────────────────────────────────
    # GAS hyperparameters
    _GAS_RHO = 0.95
    _GAS_EPS = 1e-8

    # Initialise baseline transition matrix and stationary distribution
    Pi_base = build_transition_matrix(omega, K, parameterization)
    xi_prev = stationary_distribution(Pi_base)

    # Storage
    filtered_probs = np.zeros((T, K))
    pred_probs_all = np.zeros((T, K))

    # Running state for TVTP models
    f_prev = omega.copy()

    # GAS-specific running state
    if transition_model == 'ModelIII':
        if A is None or B is None:
            raise ValueError("ModelIII requires both A and B.")
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        I_prev = np.ones(n_f)   # OPG EMA of squared scores
        s_prev = np.zeros(n_f)  # scaled score, initialised to zero

    elif transition_model in ('ModelI', 'ModelII'):
        if A is None:
            raise ValueError(f"{transition_model} requires A.")
        A = np.asarray(A, dtype=float)

    loglik = 0.0

    for t in range(T):
        # ------------------------------------------------------------------
        # 1. Compute logit parameters f_t for this period
        # ------------------------------------------------------------------
        if transition_model == 'Constant':
            f_t = omega

        elif transition_model == 'ModelI':
            if t == 0:
                f_t = omega
            else:
                f_t = omega + A * y[t - 1]

        elif transition_model == 'ModelII':
            if t == 0:
                f_t = omega
            else:
                if X is None:
                    raise ValueError("ModelII requires exogenous series X.")
                f_t = omega + A * X[t - 1]

        elif transition_model == 'ModelIII':
            if t == 0:
                f_t = omega.copy()
                # s_prev = 0 means f_1 = omega + A*0 + B*(omega-omega) = omega
            else:
                f_t = omega + A * s_prev + B * (f_prev - omega)

        else:
            raise ValueError(f"Unknown transition_model='{transition_model}'")

        # ------------------------------------------------------------------
        # 2. Build transition matrix and compute predicted probabilities
        # ------------------------------------------------------------------
        Pi_t = build_transition_matrix(f_t, K, parameterization)
        pred_probs = xi_prev @ Pi_t          # shape (K,)
        pred_probs = np.maximum(pred_probs, 0.0)
        pred_probs /= pred_probs.sum()       # renormalise for numerical safety

        # ------------------------------------------------------------------
        # 3. Observation densities and predictive density
        # ------------------------------------------------------------------
        phi_t = gaussian_pdf(y[t], mu, sigma2)
        p_t = float(pred_probs @ phi_t)
        p_t = max(p_t, 1e-300)

        # ------------------------------------------------------------------
        # 4. Updated (filtered) probabilities
        # ------------------------------------------------------------------
        xi_t = (pred_probs * phi_t) / p_t
        xi_t = np.maximum(xi_t, 0.0)
        xi_t /= xi_t.sum()

        # ------------------------------------------------------------------
        # 5. Log-likelihood accumulation
        # ------------------------------------------------------------------
        if burn_in <= t < T - end_trim:
            loglik += np.log(p_t)

        # ------------------------------------------------------------------
        # 6. GAS score update (needed for next period's f)
        # ------------------------------------------------------------------
        if transition_model == 'ModelIII':
            nabla = _gas_score(xi_prev, phi_t, Pi_t, p_t, K, parameterization)
            # OPG EMA Fisher info
            I_t = _GAS_RHO * I_prev + (1.0 - _GAS_RHO) * nabla ** 2
            s_t = nabla / np.sqrt(I_t + _GAS_EPS)
            # Carry forward for next iteration
            I_prev = I_t
            s_prev = s_t

        # ------------------------------------------------------------------
        # 7. Store results and advance state
        # ------------------------------------------------------------------
        filtered_probs[t] = xi_t
        pred_probs_all[t] = pred_probs
        xi_prev = xi_t
        f_prev = f_t

    return loglik, filtered_probs, pred_probs_all
