"""
Numba-JIT Hamilton filter kernels.

Provides three JIT-compiled filter loops covering all (K, parameterization) cases
used in the study.  Each kernel handles all four transition models (Constant,
ModelI, ModelII, ModelIII) via an integer dispatch argument.

Model-type integer codes
  0 = Constant
  1 = ModelI   (logit driven by lagged y_{t-1})
  2 = ModelII  (logit driven by lagged exogenous X_{t-1})
  3 = ModelIII (GAS/score-driven with OPG-EMA Fisher scaling)

Custom implementation — Context7 found no library equivalent for the
combined TVTP-GAS Markov-switching filter with off-diagonal row rescaling
and paper-exact likelihood trimming (Bazzi et al., 2017 GAS mechanics).
"""

import numpy as np
import numba

# ---------------------------------------------------------------------------
# GAS hyperparameters (mirrored from filter.py)
# ---------------------------------------------------------------------------
_GAS_RHO = 0.95
_GAS_EPS = 1e-8

# ---------------------------------------------------------------------------
# Numba scalar helpers
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _sig(x: float) -> float:
    """Numerically stable sigmoid (avoids overflow for large |x|)."""
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


@numba.njit(cache=True)
def _gauss(yt: float, mu: float, sig2: float) -> float:
    """Gaussian density N(yt; mu, sig2), clipped at 1e-300."""
    diff = yt - mu
    v = np.exp(-0.5 * diff * diff / sig2) / (2.5066282746310002 * np.sqrt(sig2))
    if v < 1e-300:
        v = 1e-300
    return v


# ---------------------------------------------------------------------------
# K=2, DIAGONAL parameterisation
#   omega / A / B  :  length-2 arrays  [f_11, f_22]
#   pi11 = sig(f11),  pi12 = 1 - pi11
#   pi22 = sig(f22),  pi21 = 1 - pi22
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def nb_filter_K2_diag(
    y: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    omega: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    model_type: int,
    burn_in: int,
    end_trim: int,
) -> tuple:
    """
    Hamilton filter for K=2, diagonal parameterisation.

    Parameters
    ----------
    y          : observed series (T,)
    mu         : regime means (2,)
    sigma2     : regime variances (2,)
    omega      : baseline logit parameters [f_11, f_22]  (2,)
    A          : TVTP score coefficients (2,)   — zeros for Constant
    B          : GAS persistence coefficients (2,) — zeros for non-GAS
    X          : exogenous driver (T,) — zeros for non-ModelII
    model_type : 0=Constant  1=ModelI  2=ModelII  3=ModelIII
    burn_in    : first `burn_in` time steps excluded from log-likelihood
    end_trim   : last  `end_trim` time steps excluded from log-likelihood

    Returns
    -------
    loglik     : accumulated log-likelihood over [burn_in, T-end_trim)
    filtered   : (T, 2) filtered regime probabilities
    predicted  : (T, 2) one-step-ahead predicted probabilities
    """
    T = len(y)
    GAS_RHO = 0.95
    GAS_EPS = 1e-8

    # Stationary initialisation from baseline transition matrix
    p12_b = 1.0 - _sig(omega[0])
    p21_b = 1.0 - _sig(omega[1])
    denom = p12_b + p21_b
    if denom < 1e-15:
        denom = 1e-15
    xi0 = p21_b / denom
    xi1 = p12_b / denom

    # GAS state — f holds logit parameters (updated in place)
    f0 = omega[0]
    f1 = omega[1]
    I0 = 1.0;  I1 = 1.0   # OPG EMA of squared scores
    s0_prev = 0.0;  s1_prev = 0.0

    loglik = 0.0
    filtered = np.empty((T, 2))
    predicted = np.empty((T, 2))

    for t in range(T):
        xi0_prev = xi0;  xi1_prev = xi1

        # ── 1. Logit parameters f_t ──────────────────────────────────────
        if model_type == 0:        # Constant
            f0 = omega[0];  f1 = omega[1]
        elif model_type == 1:      # ModelI: lagged y
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * y[t - 1]
                f1 = omega[1] + A[1] * y[t - 1]
        elif model_type == 2:      # ModelII: lagged X
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * X[t - 1]
                f1 = omega[1] + A[1] * X[t - 1]
        else:                      # ModelIII: GAS
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * s0_prev + B[0] * (f0 - omega[0])
                f1 = omega[1] + A[1] * s1_prev + B[1] * (f1 - omega[1])

        # ── 2. Transition matrix ─────────────────────────────────────────
        p11 = _sig(f0);  p12 = 1.0 - p11
        p22 = _sig(f1);  p21 = 1.0 - p22

        # ── 3. Predicted probabilities  (xi_prev @ Pi) ──────────────────
        pr0 = xi0_prev * p11 + xi1_prev * p21
        pr1 = xi0_prev * p12 + xi1_prev * p22
        pr_s = pr0 + pr1
        if pr_s < 1e-15:
            pr_s = 1e-15
        pr0 /= pr_s;  pr1 /= pr_s
        predicted[t, 0] = pr0;  predicted[t, 1] = pr1

        # ── 4. Gaussian densities ────────────────────────────────────────
        phi0 = _gauss(y[t], mu[0], sigma2[0])
        phi1 = _gauss(y[t], mu[1], sigma2[1])

        # ── 5. Predictive density ────────────────────────────────────────
        p_t = pr0 * phi0 + pr1 * phi1
        if p_t < 1e-300:
            p_t = 1e-300

        # ── 6. Filtered probabilities ────────────────────────────────────
        xi0 = pr0 * phi0 / p_t
        xi1 = pr1 * phi1 / p_t
        xi_s = xi0 + xi1
        if xi_s < 1e-15:
            xi_s = 1e-15
        xi0 /= xi_s;  xi1 /= xi_s
        filtered[t, 0] = xi0;  filtered[t, 1] = xi1

        # ── 7. Log-likelihood accumulation ──────────────────────────────
        if burn_in <= t < T - end_trim:
            loglik += np.log(p_t)

        # ── 8. GAS score update  (used at t+1) ──────────────────────────
        if model_type == 3:
            sp11 = p11 * p12          # pi11 * (1 - pi11)
            sp22 = p22 * p21          # pi22 * (1 - pi22)
            nabla0 = xi0_prev * sp11 * (phi0 - phi1) / p_t
            nabla1 = xi1_prev * sp22 * (phi1 - phi0) / p_t
            I0 = GAS_RHO * I0 + (1.0 - GAS_RHO) * nabla0 * nabla0
            I1 = GAS_RHO * I1 + (1.0 - GAS_RHO) * nabla1 * nabla1
            s0_prev = nabla0 / np.sqrt(I0 + GAS_EPS)
            s1_prev = nabla1 / np.sqrt(I1 + GAS_EPS)

    return loglik, filtered, predicted


# ---------------------------------------------------------------------------
# K=2, OFF_DIAGONAL parameterisation
#   omega / A / B  :  length-2 arrays  [f_12, f_21]
#   pi12 = sig(f12),  pi11 = 1 - pi12
#   pi21 = sig(f21),  pi22 = 1 - pi21
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def nb_filter_K2_offdiag(
    y: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    omega: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    model_type: int,
    burn_in: int,
    end_trim: int,
) -> tuple:
    """Hamilton filter for K=2, off_diagonal parameterisation.  [f_12, f_21]"""
    T = len(y)
    GAS_RHO = 0.95
    GAS_EPS = 1e-8

    p12_b = _sig(omega[0]);  p21_b = _sig(omega[1])
    denom = p12_b + p21_b
    if denom < 1e-15:
        denom = 1e-15
    xi0 = p21_b / denom
    xi1 = p12_b / denom

    f0 = omega[0];  f1 = omega[1]
    I0 = 1.0;  I1 = 1.0
    s0_prev = 0.0;  s1_prev = 0.0

    loglik = 0.0
    filtered = np.empty((T, 2))
    predicted = np.empty((T, 2))

    for t in range(T):
        xi0_prev = xi0;  xi1_prev = xi1

        if model_type == 0:
            f0 = omega[0];  f1 = omega[1]
        elif model_type == 1:
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * y[t - 1]
                f1 = omega[1] + A[1] * y[t - 1]
        elif model_type == 2:
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * X[t - 1]
                f1 = omega[1] + A[1] * X[t - 1]
        else:  # GAS
            if t == 0:
                f0 = omega[0];  f1 = omega[1]
            else:
                f0 = omega[0] + A[0] * s0_prev + B[0] * (f0 - omega[0])
                f1 = omega[1] + A[1] * s1_prev + B[1] * (f1 - omega[1])

        p12 = _sig(f0);  p11 = 1.0 - p12
        p21 = _sig(f1);  p22 = 1.0 - p21

        pr0 = xi0_prev * p11 + xi1_prev * p21
        pr1 = xi0_prev * p12 + xi1_prev * p22
        pr_s = pr0 + pr1
        if pr_s < 1e-15:
            pr_s = 1e-15
        pr0 /= pr_s;  pr1 /= pr_s
        predicted[t, 0] = pr0;  predicted[t, 1] = pr1

        phi0 = _gauss(y[t], mu[0], sigma2[0])
        phi1 = _gauss(y[t], mu[1], sigma2[1])

        p_t = pr0 * phi0 + pr1 * phi1
        if p_t < 1e-300:
            p_t = 1e-300

        xi0 = pr0 * phi0 / p_t
        xi1 = pr1 * phi1 / p_t
        xi_s = xi0 + xi1
        if xi_s < 1e-15:
            xi_s = 1e-15
        xi0 /= xi_s;  xi1 /= xi_s
        filtered[t, 0] = xi0;  filtered[t, 1] = xi1

        if burn_in <= t < T - end_trim:
            loglik += np.log(p_t)

        if model_type == 3:
            sp12 = p12 * p11          # pi12 * (1 - pi12)
            sp21 = p21 * p22          # pi21 * (1 - pi21)
            nabla0 = xi0_prev * sp12 * (phi1 - phi0) / p_t
            nabla1 = xi1_prev * sp21 * (phi0 - phi1) / p_t
            I0 = GAS_RHO * I0 + (1.0 - GAS_RHO) * nabla0 * nabla0
            I1 = GAS_RHO * I1 + (1.0 - GAS_RHO) * nabla1 * nabla1
            s0_prev = nabla0 / np.sqrt(I0 + GAS_EPS)
            s1_prev = nabla1 / np.sqrt(I1 + GAS_EPS)

    return loglik, filtered, predicted


# ---------------------------------------------------------------------------
# K=3, OFF_DIAGONAL parameterisation
#   omega / A / B  :  length-6 arrays  [f_12, f_13, f_21, f_23, f_31, f_32]
#   Row i : pi_{ij1} = sig(f[2i]), pi_{ij2} = sig(f[2i+1]);
#            rescale if row sum > 1;  pi_{ii} = 1 - sum_offdiag
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def nb_filter_K3_offdiag(
    y: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    omega: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    model_type: int,
    burn_in: int,
    end_trim: int,
) -> tuple:
    """
    Hamilton filter for K=3, off_diagonal parameterisation.
    omega[0..5] = [f_12, f_13, f_21, f_23, f_31, f_32]

    Off-diagonal row sum > 1 triggers proportional rescaling.
    GAS score uses the unrescaled logistic derivative (rescale correction ≈ 1
    when off-diagonal probabilities are small, as in all study DGPs).
    """
    T = len(y)
    GAS_RHO = 0.95
    GAS_EPS = 1e-8

    # ── Stationary distribution via power iteration on baseline Pi ──────
    p12_b = _sig(omega[0]);  p13_b = _sig(omega[1])
    s0_b  = p12_b + p13_b
    if s0_b > 1.0 - 1e-10:
        p12_b /= (s0_b + 1e-10);  p13_b /= (s0_b + 1e-10);  s0_b = p12_b + p13_b
    p11_b = 0.0 if 1.0 - s0_b < 0.0 else 1.0 - s0_b

    p21_b = _sig(omega[2]);  p23_b = _sig(omega[3])
    s1_b  = p21_b + p23_b
    if s1_b > 1.0 - 1e-10:
        p21_b /= (s1_b + 1e-10);  p23_b /= (s1_b + 1e-10);  s1_b = p21_b + p23_b
    p22_b = 0.0 if 1.0 - s1_b < 0.0 else 1.0 - s1_b

    p31_b = _sig(omega[4]);  p32_b = _sig(omega[5])
    s2_b  = p31_b + p32_b
    if s2_b > 1.0 - 1e-10:
        p31_b /= (s2_b + 1e-10);  p32_b /= (s2_b + 1e-10);  s2_b = p31_b + p32_b
    p33_b = 0.0 if 1.0 - s2_b < 0.0 else 1.0 - s2_b

    xi0 = 1.0 / 3.0;  xi1 = 1.0 / 3.0;  xi2 = 1.0 / 3.0
    for _ in range(500):
        n0 = xi0 * p11_b + xi1 * p21_b + xi2 * p31_b
        n1 = xi0 * p12_b + xi1 * p22_b + xi2 * p32_b
        n2 = xi0 * p13_b + xi1 * p23_b + xi2 * p33_b
        sn = n0 + n1 + n2
        if sn < 1e-15:
            sn = 1e-15
        n0 /= sn;  n1 /= sn;  n2 /= sn
        if abs(n0 - xi0) + abs(n1 - xi1) + abs(n2 - xi2) < 1e-10:
            xi0 = n0;  xi1 = n1;  xi2 = n2
            break
        xi0 = n0;  xi1 = n1;  xi2 = n2

    # ── GAS state — f[0..5] updated in place each step ──────────────────
    f    = np.empty(6)
    I_g  = np.ones(6)
    s_p  = np.zeros(6)
    for k in range(6):
        f[k] = omega[k]

    loglik   = 0.0
    filtered  = np.empty((T, 3))
    predicted = np.empty((T, 3))

    for t in range(T):
        xi0_prev = xi0;  xi1_prev = xi1;  xi2_prev = xi2

        # ── 1. Logit parameters f_t ──────────────────────────────────────
        if model_type == 0:
            for k in range(6):
                f[k] = omega[k]
        elif model_type == 1:
            if t == 0:
                for k in range(6):
                    f[k] = omega[k]
            else:
                yt1 = y[t - 1]
                for k in range(6):
                    f[k] = omega[k] + A[k] * yt1
        elif model_type == 2:
            if t == 0:
                for k in range(6):
                    f[k] = omega[k]
            else:
                xt1 = X[t - 1]
                for k in range(6):
                    f[k] = omega[k] + A[k] * xt1
        else:  # GAS
            if t == 0:
                for k in range(6):
                    f[k] = omega[k]
            else:
                for k in range(6):
                    # f[k] still holds f_{t-1} at this point
                    f[k] = omega[k] + A[k] * s_p[k] + B[k] * (f[k] - omega[k])

        # ── 2. Transition matrix (in-register scalars) ───────────────────
        p12 = _sig(f[0]);  p13 = _sig(f[1])
        sr0 = p12 + p13
        if sr0 > 1.0 - 1e-10:
            p12 /= (sr0 + 1e-10);  p13 /= (sr0 + 1e-10);  sr0 = p12 + p13
        p11 = 0.0 if 1.0 - sr0 < 0.0 else 1.0 - sr0

        p21 = _sig(f[2]);  p23 = _sig(f[3])
        sr1 = p21 + p23
        if sr1 > 1.0 - 1e-10:
            p21 /= (sr1 + 1e-10);  p23 /= (sr1 + 1e-10);  sr1 = p21 + p23
        p22 = 0.0 if 1.0 - sr1 < 0.0 else 1.0 - sr1

        p31 = _sig(f[4]);  p32 = _sig(f[5])
        sr2 = p31 + p32
        if sr2 > 1.0 - 1e-10:
            p31 /= (sr2 + 1e-10);  p32 /= (sr2 + 1e-10);  sr2 = p31 + p32
        p33 = 0.0 if 1.0 - sr2 < 0.0 else 1.0 - sr2

        # ── 3. Predicted probabilities ───────────────────────────────────
        pr0 = xi0_prev * p11 + xi1_prev * p21 + xi2_prev * p31
        pr1 = xi0_prev * p12 + xi1_prev * p22 + xi2_prev * p32
        pr2 = xi0_prev * p13 + xi1_prev * p23 + xi2_prev * p33
        prs = pr0 + pr1 + pr2
        if prs < 1e-15:
            prs = 1e-15
        pr0 /= prs;  pr1 /= prs;  pr2 /= prs
        predicted[t, 0] = pr0;  predicted[t, 1] = pr1;  predicted[t, 2] = pr2

        # ── 4. Gaussian densities ────────────────────────────────────────
        phi0 = _gauss(y[t], mu[0], sigma2[0])
        phi1 = _gauss(y[t], mu[1], sigma2[1])
        phi2 = _gauss(y[t], mu[2], sigma2[2])

        # ── 5. Predictive density ────────────────────────────────────────
        p_t = pr0 * phi0 + pr1 * phi1 + pr2 * phi2
        if p_t < 1e-300:
            p_t = 1e-300

        # ── 6. Filtered probabilities ────────────────────────────────────
        xi0 = pr0 * phi0 / p_t
        xi1 = pr1 * phi1 / p_t
        xi2 = pr2 * phi2 / p_t
        xis = xi0 + xi1 + xi2
        if xis < 1e-15:
            xis = 1e-15
        xi0 /= xis;  xi1 /= xis;  xi2 /= xis
        filtered[t, 0] = xi0;  filtered[t, 1] = xi1;  filtered[t, 2] = xi2

        # ── 7. Log-likelihood ────────────────────────────────────────────
        if burn_in <= t < T - end_trim:
            loglik += np.log(p_t)

        # ── 8. GAS score update (unrescaled logistic derivative) ─────────
        if model_type == 3:
            sp12 = p12 * (1.0 - p12)
            sp13 = p13 * (1.0 - p13)
            sp21 = p21 * (1.0 - p21)
            sp23 = p23 * (1.0 - p23)
            sp31 = p31 * (1.0 - p31)
            sp32 = p32 * (1.0 - p32)
            inv_pt = 1.0 / p_t

            nb0 = xi0_prev * sp12 * (phi1 - phi0) * inv_pt   # df12
            nb1 = xi0_prev * sp13 * (phi2 - phi0) * inv_pt   # df13
            nb2 = xi1_prev * sp21 * (phi0 - phi1) * inv_pt   # df21
            nb3 = xi1_prev * sp23 * (phi2 - phi1) * inv_pt   # df23
            nb4 = xi2_prev * sp31 * (phi0 - phi2) * inv_pt   # df31
            nb5 = xi2_prev * sp32 * (phi1 - phi2) * inv_pt   # df32

            I_g[0] = GAS_RHO * I_g[0] + (1.0 - GAS_RHO) * nb0 * nb0
            I_g[1] = GAS_RHO * I_g[1] + (1.0 - GAS_RHO) * nb1 * nb1
            I_g[2] = GAS_RHO * I_g[2] + (1.0 - GAS_RHO) * nb2 * nb2
            I_g[3] = GAS_RHO * I_g[3] + (1.0 - GAS_RHO) * nb3 * nb3
            I_g[4] = GAS_RHO * I_g[4] + (1.0 - GAS_RHO) * nb4 * nb4
            I_g[5] = GAS_RHO * I_g[5] + (1.0 - GAS_RHO) * nb5 * nb5

            s_p[0] = nb0 / np.sqrt(I_g[0] + GAS_EPS)
            s_p[1] = nb1 / np.sqrt(I_g[1] + GAS_EPS)
            s_p[2] = nb2 / np.sqrt(I_g[2] + GAS_EPS)
            s_p[3] = nb3 / np.sqrt(I_g[3] + GAS_EPS)
            s_p[4] = nb4 / np.sqrt(I_g[4] + GAS_EPS)
            s_p[5] = nb5 / np.sqrt(I_g[5] + GAS_EPS)

    return loglik, filtered, predicted


# ---------------------------------------------------------------------------
# Warmup — trigger JIT compilation at import time with minimal dummy data
# ---------------------------------------------------------------------------

def warmup() -> None:
    """
    JIT-compile all three filter kernels by running them on tiny dummy data.
    Call once at startup; subsequent calls use the cached compiled functions.
    """
    _y     = np.array([0.0, 0.1, -0.1])
    _mu2   = np.array([-1.0, 1.0])
    _sig2  = np.array([0.5, 0.5])
    _mu3   = np.array([-2.0, 0.0, 2.0])
    _sig3  = np.array([0.3, 0.5, 0.8])
    _om2   = np.array([1.0, 2.0])
    _om3   = np.array([-2.4, -2.4, -2.2, -2.2, -2.6, -2.6])
    _Z2    = np.zeros(2);  _Z3 = np.zeros(6)
    _Xt    = np.zeros(3)

    nb_filter_K2_diag(_y, _mu2, _sig2, _om2, _Z2, _Z2, _Xt, 0, 0, 0)
    nb_filter_K2_offdiag(_y, _mu2, _sig2, _om2, _Z2, _Z2, _Xt, 0, 0, 0)
    nb_filter_K3_offdiag(_y, _mu3, _sig3, _om3, _Z3, _Z3, _Xt, 0, 0, 0)
