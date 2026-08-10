"""
Simulation of Markov-switching data-generating processes.

Supports all four transition specifications (Constant, ModelI, ModelII,
ModelIII/GAS) with diagonal and off-diagonal parameterisations for K=2 and
K=3 regimes.
"""

import numpy as np
from typing import Optional, Tuple

from config.dgp_params import DGPConfig
from src.filter import (
    build_transition_matrix,
    stationary_distribution,
    gaussian_pdf,
    sigmoid,
    _gas_score,
)


def simulate_dgp(
    dgp: DGPConfig,
    T: int,
    rng: np.random.Generator,
    X_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Simulate a Markov-switching time series from a given DGP.

    Parameters
    ----------
    dgp    : DGP configuration (true parameters)
    T      : number of observations to generate
    rng    : NumPy random Generator (for reproducibility)
    X_sigma: standard deviation for the iid exogenous driver X_t (ModelII)

    Returns
    -------
    y : (T,) observed series
    z : (T,) latent regime indices, 0-indexed
    X : (T,) exogenous series if ModelII, else None

    Notes
    -----
    Initialization: the latent state at t=0 is drawn from the stationary
    distribution of the *baseline* transition matrix (Pi evaluated at omega).

    For ModelIII (GAS), the same OPG EMA score recursion used in estimation
    is applied during simulation, ensuring that the simulated data are
    consistent with the estimated model.
    """
    K = dgp.K
    mu = np.array(dgp.mu, dtype=float)
    sigma2 = dgp.get_sigma2_full()
    omega = dgp.get_omega()
    parameterization = dgp.parameterization
    transition_model = dgp.transition_model

    A = np.array(dgp.A, dtype=float) if dgp.A is not None else None
    B = np.array(dgp.B, dtype=float) if dgp.B is not None else None

    n_f = len(omega)

    # GAS hyperparameters (must match filter.py)
    _GAS_RHO = 0.95
    _GAS_EPS = 1e-8

    # Precompute baseline matrix and stationary distribution
    Pi_base = build_transition_matrix(omega, K, parameterization)
    pi_stat = stationary_distribution(Pi_base)

    # Allocate output arrays
    y = np.zeros(T, dtype=float)
    z = np.zeros(T, dtype=int)
    X = rng.normal(0.0, X_sigma, size=T) if transition_model == 'ModelII' else None

    # Draw initial state
    z[0] = int(rng.choice(K, p=pi_stat))
    y[0] = float(rng.normal(mu[z[0]], np.sqrt(sigma2[z[0]])))

    # GAS running state
    f_prev = omega.copy()
    I_prev = np.ones(n_f)
    s_prev = np.zeros(n_f)

    # If ModelIII, we need the filter state xi_prev (for score computation)
    if transition_model == 'ModelIII':
        xi_prev = pi_stat.copy()
        # Update xi after first observation
        Pi_0 = build_transition_matrix(omega, K, parameterization)
        pred_0 = xi_prev @ Pi_0
        pred_0 = np.maximum(pred_0, 0.0)
        pred_0 /= pred_0.sum()
        phi_0 = gaussian_pdf(y[0], mu, sigma2)
        p_0 = max(float(pred_0 @ phi_0), 1e-300)
        xi_0 = (pred_0 * phi_0) / p_0
        xi_0 = np.maximum(xi_0, 0.0)
        xi_0 /= xi_0.sum()
        # Compute score at t=0
        nabla_0 = _gas_score(xi_prev, phi_0, Pi_0, p_0, K, parameterization)
        I_prev = _GAS_RHO * I_prev + (1.0 - _GAS_RHO) * nabla_0 ** 2
        s_prev = nabla_0 / np.sqrt(I_prev + _GAS_EPS)
        f_prev = omega.copy()   # f_1 = omega + A*s_0 + B*(f_0-omega) computed below
        xi_prev = xi_0

    for t in range(1, T):
        # Compute logit parameters for this period
        if transition_model == 'Constant':
            f_t = omega

        elif transition_model == 'ModelI':
            f_t = omega + A * y[t - 1]

        elif transition_model == 'ModelII':
            f_t = omega + A * X[t - 1]

        elif transition_model == 'ModelIII':
            f_t = omega + A * s_prev + B * (f_prev - omega)

        else:
            raise ValueError(f"Unknown transition_model='{transition_model}'")

        # Build transition matrix and sample next state
        Pi_t = build_transition_matrix(f_t, K, parameterization)
        probs = Pi_t[z[t - 1], :]
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        z[t] = int(rng.choice(K, p=probs))

        # Sample observation
        y[t] = float(rng.normal(mu[z[t]], np.sqrt(sigma2[z[t]])))

        # Update GAS state for next iteration
        if transition_model == 'ModelIII':
            pred_t = xi_prev @ Pi_t
            pred_t = np.maximum(pred_t, 0.0)
            pred_t /= pred_t.sum()
            phi_t = gaussian_pdf(y[t], mu, sigma2)
            p_t = max(float(pred_t @ phi_t), 1e-300)
            xi_t = (pred_t * phi_t) / p_t
            xi_t = np.maximum(xi_t, 0.0)
            xi_t /= xi_t.sum()

            nabla_t = _gas_score(xi_prev, phi_t, Pi_t, p_t, K, parameterization)
            I_prev = _GAS_RHO * I_prev + (1.0 - _GAS_RHO) * nabla_t ** 2
            s_prev = nabla_t / np.sqrt(I_prev + _GAS_EPS)
            xi_prev = xi_t

        f_prev = f_t

    return y, z, X
