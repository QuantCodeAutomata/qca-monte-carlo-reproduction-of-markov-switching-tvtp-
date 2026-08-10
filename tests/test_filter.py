"""Tests for src/filter.py — build_transition_matrix and hamilton_filter."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

def test_transition_matrix_k2_diagonal_rows_sum_to_one():
    from src.filter import build_transition_matrix
    f = np.array([1.0, 0.5])  # logit scale
    Pi = build_transition_matrix(f, K=2, parameterization='diagonal')
    assert Pi.shape == (2, 2)
    np.testing.assert_allclose(Pi.sum(axis=1), np.ones(2), atol=1e-10)
    assert np.all(Pi >= 0)


def test_transition_matrix_k2_diagonal_values_correct():
    from src.filter import build_transition_matrix
    from scipy.special import expit
    f = np.array([1.0, 0.5])
    Pi = build_transition_matrix(f, K=2, parameterization='diagonal')
    # p11 = sigmoid(f[0]), p22 = sigmoid(f[1])
    np.testing.assert_allclose(Pi[0, 0], float(expit(1.0)), atol=1e-10)
    np.testing.assert_allclose(Pi[1, 1], float(expit(0.5)), atol=1e-10)
    np.testing.assert_allclose(Pi[0, 1], 1.0 - float(expit(1.0)), atol=1e-10)


def test_transition_matrix_k3_offdiag_rows_sum_to_one():
    from src.filter import build_transition_matrix
    f = np.array([-2.5, -2.5, -2.4, -2.4, -2.6, -2.6])  # small off-diag probs
    Pi = build_transition_matrix(f, K=3, parameterization='off_diagonal')
    assert Pi.shape == (3, 3)
    np.testing.assert_allclose(Pi.sum(axis=1), np.ones(3), atol=1e-9)
    assert np.all(Pi >= 0)
    assert np.all(Pi <= 1)


def test_transition_matrix_k3_offdiag_row_rescaling():
    from src.filter import build_transition_matrix
    # Very large logits → sigmoid ≈ 1 for each off-diagonal → row sum > 1
    f = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    Pi = build_transition_matrix(f, K=3, parameterization='off_diagonal')
    np.testing.assert_allclose(Pi.sum(axis=1), np.ones(3), atol=1e-9)
    assert np.all(Pi >= 0)
    assert np.all(Pi <= 1)


def test_transition_matrix_unsupported_raises():
    from src.filter import build_transition_matrix
    with pytest.raises(ValueError):
        build_transition_matrix(np.zeros(4), K=4, parameterization='diagonal')


# ---------------------------------------------------------------------------
# hamilton_filter — Constant model
# ---------------------------------------------------------------------------

def test_hamilton_filter_constant_loglik_finite():
    from src.filter import hamilton_filter
    np.random.seed(0)
    y = np.random.randn(200)
    mu = np.array([-1.0, 1.0])
    sigma2 = np.array([0.5, 0.5])
    omega = np.array([1.386, 2.197])   # logit(0.8), logit(0.9)
    ll, filt, pred = hamilton_filter(
        y, mu, sigma2, K=2,
        parameterization='diagonal', transition_model='Constant',
        omega=omega, burn_in=10, end_trim=0,
    )
    assert np.isfinite(ll)
    assert filt.shape == (200, 2)
    np.testing.assert_allclose(filt.sum(axis=1), np.ones(200), atol=1e-9)


def test_hamilton_filter_returns_three_elements():
    from src.filter import hamilton_filter
    y = np.random.randn(100)
    mu = np.array([-1.0, 1.0])
    sigma2 = np.array([0.5, 0.5])
    omega = np.array([1.0, 1.0])
    result = hamilton_filter(
        y, mu, sigma2, 2, 'diagonal', 'Constant',
        omega, burn_in=0, end_trim=0,
    )
    assert len(result) == 3
    ll, filt, pred = result
    assert np.isfinite(ll)


def test_filtered_probs_between_zero_and_one():
    from src.filter import hamilton_filter
    np.random.seed(7)
    y = np.random.randn(500)
    mu = np.array([-2.0, 0.0, 2.0])
    sigma2 = np.array([0.3, 0.5, 0.8])
    omega = np.array([
        np.log(0.08 / 0.92), np.log(0.08 / 0.92),
        np.log(0.10 / 0.90), np.log(0.10 / 0.90),
        np.log(0.06 / 0.94), np.log(0.06 / 0.94),
    ])
    ll, filt, pred = hamilton_filter(
        y, mu, sigma2, K=3,
        parameterization='off_diagonal', transition_model='Constant',
        omega=omega, burn_in=10, end_trim=0,
    )
    assert np.all(filt >= -1e-10)
    assert np.all(filt <= 1 + 1e-10)
    np.testing.assert_allclose(filt.sum(axis=1), np.ones(500), atol=1e-9)


def test_pred_probs_between_zero_and_one():
    from src.filter import hamilton_filter
    np.random.seed(3)
    y = np.random.randn(300)
    mu = np.array([-1.0, 1.0])
    sigma2 = np.array([1.0, 1.0])
    omega = np.array([1.386, 1.386])
    _, filt, pred = hamilton_filter(
        y, mu, sigma2, 2, 'diagonal', 'Constant',
        omega, burn_in=5, end_trim=0,
    )
    assert np.all(pred >= -1e-10)
    assert np.all(pred <= 1 + 1e-10)
    np.testing.assert_allclose(pred.sum(axis=1), np.ones(300), atol=1e-9)


# ---------------------------------------------------------------------------
# hamilton_filter — GAS collapse test
# ---------------------------------------------------------------------------

def test_gas_a_zero_equals_constant():
    """Model III with A=0 should give identical log-likelihood as Constant."""
    from src.filter import hamilton_filter
    np.random.seed(123)
    y = np.random.randn(300)
    mu = np.array([-1.0, 1.0])
    sigma2 = np.array([0.5, 0.5])
    omega = np.array([1.386, 2.197])
    A_zero = np.zeros(2)
    B = np.array([0.9, 0.85])

    ll_const, filt_const, _ = hamilton_filter(
        y, mu, sigma2, K=2,
        parameterization='diagonal', transition_model='Constant',
        omega=omega, burn_in=10, end_trim=0,
    )
    ll_gas, filt_gas, _ = hamilton_filter(
        y, mu, sigma2, K=2,
        parameterization='diagonal', transition_model='ModelIII',
        omega=omega, A=A_zero, B=B, burn_in=10, end_trim=0,
    )

    # When A=0: f_t = omega + 0*s + B*(omega-omega) = omega → identical to Constant
    np.testing.assert_allclose(ll_const, ll_gas, atol=1e-5)


# ---------------------------------------------------------------------------
# hamilton_filter — burn-in / end-trim
# ---------------------------------------------------------------------------

def test_effective_loglik_sample_sizes():
    """Loglik should be finite for T=500 and T=1000 with standard trimming."""
    from src.filter import hamilton_filter
    from config.settings import BURN_IN, END_TRIM

    for T in [500, 1000]:
        np.random.seed(T)
        y = np.random.randn(T)
        mu = np.array([-1.0, 1.0])
        sigma2 = np.array([0.5, 0.5])
        omega = np.array([1.386, 2.197])
        ll, _, _ = hamilton_filter(
            y, mu, sigma2, K=2, parameterization='diagonal',
            transition_model='Constant', omega=omega,
            burn_in=BURN_IN, end_trim=END_TRIM,
        )
        assert np.isfinite(ll), f'T={T}: loglik should be finite'


def test_loglik_increases_with_better_params():
    """A well-specified Constant model should achieve higher LL than uniform one."""
    from src.filter import hamilton_filter
    np.random.seed(42)
    # Generate from a clear 2-regime model
    T = 1000
    z = np.where(np.arange(T) % 20 < 10, 0, 1)
    y = np.where(z == 0, np.random.normal(-2, 0.3, T), np.random.normal(2, 0.3, T))

    omega_good = np.array([2.197, 2.197])   # logit(0.9): persistent regimes
    omega_bad  = np.array([0.0, 0.0])       # logit(0.5): random switching

    ll_good, _, _ = hamilton_filter(y, np.array([-2.0, 2.0]), np.array([0.09, 0.09]),
                                    2, 'diagonal', 'Constant', omega_good,
                                    burn_in=0, end_trim=0)
    ll_bad,  _, _ = hamilton_filter(y, np.array([0.0, 0.0]),  np.array([0.5,  0.5]),
                                    2, 'diagonal', 'Constant', omega_bad,
                                    burn_in=0, end_trim=0)
    assert ll_good > ll_bad


# ---------------------------------------------------------------------------
# hamilton_filter — ModelI (TVTP with lagged y)
# ---------------------------------------------------------------------------

def test_hamilton_filter_model_i_finite():
    """ModelI filter should return finite loglik."""
    from src.filter import hamilton_filter
    np.random.seed(11)
    y = np.random.randn(200)
    mu = np.array([-1.0, 1.0])
    sigma2 = np.array([0.5, 0.5])
    omega = np.array([1.386, 2.197])
    A = np.array([0.1, -0.1])
    ll, filt, pred = hamilton_filter(
        y, mu, sigma2, 2, 'diagonal', 'ModelI',
        omega, A=A, burn_in=10, end_trim=0,
    )
    assert np.isfinite(ll)
    np.testing.assert_allclose(filt.sum(axis=1), np.ones(200), atol=1e-9)


# ---------------------------------------------------------------------------
# stationary_distribution
# ---------------------------------------------------------------------------

def test_stationary_distribution_sums_to_one():
    from src.filter import stationary_distribution, build_transition_matrix
    omega = np.array([1.5, 1.8])
    Pi = build_transition_matrix(omega, K=2, parameterization='diagonal')
    pi = stationary_distribution(Pi)
    assert pi.shape == (2,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)
    assert np.all(pi >= 0)
