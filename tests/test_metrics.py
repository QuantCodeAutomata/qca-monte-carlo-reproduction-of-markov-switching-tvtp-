"""Tests for src/metrics.py — compute_parameter_metrics, forecast/filter metrics."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# compute_parameter_metrics
# ---------------------------------------------------------------------------

def test_parameter_metrics_coverage_known():
    """Coverage should be ≈95 % when SE matches true sampling error exactly."""
    from src.metrics import compute_parameter_metrics
    rng = np.random.default_rng(100)
    true_val = np.array([1.0])
    # Estimates drawn from N(1.0, 0.1); SE fixed at 0.1 → nominal 95% CI
    estimates = [rng.normal(1.0, 0.1, 1) for _ in range(200)]
    ses = [np.array([0.1]) for _ in range(200)]
    result = compute_parameter_metrics(estimates, true_val, ses)
    assert 'coverage' in result
    assert 0.85 < result['coverage'][0] < 1.0, (
        f"Coverage {result['coverage'][0]:.3f} not in (0.85, 1.0)"
    )


def test_bias_rmse_known_values():
    """Bias and RMSE should equal 0.1 when all estimates are offset by +0.1."""
    from src.metrics import compute_parameter_metrics
    true_val = np.array([1.0, 2.0])
    estimates = [np.array([1.1, 2.1]) for _ in range(100)]
    ses = [np.array([0.5, 0.5]) for _ in range(100)]
    result = compute_parameter_metrics(estimates, true_val, ses)
    np.testing.assert_allclose(result['bias'], [0.1, 0.1], atol=1e-10)
    np.testing.assert_allclose(result['rmse'], [0.1, 0.1], atol=1e-10)


def test_bias_zero_for_unbiased_estimates():
    """Bias should be ≈0 for large enough sample of centred estimates."""
    from src.metrics import compute_parameter_metrics
    rng = np.random.default_rng(7)
    true_val = np.array([0.0, 3.0])
    R = 500
    estimates = [true_val + rng.normal(0, 0.1, 2) for _ in range(R)]
    ses = [np.full(2, 0.1) for _ in range(R)]
    result = compute_parameter_metrics(estimates, true_val, ses)
    np.testing.assert_allclose(result['bias'], [0.0, 0.0], atol=0.05)


def test_parameter_metrics_returns_expected_keys():
    from src.metrics import compute_parameter_metrics
    estimates = [np.array([1.0]) for _ in range(10)]
    ses = [np.array([0.1]) for _ in range(10)]
    result = compute_parameter_metrics(estimates, np.array([1.0]), ses)
    for key in ('bias', 'rmse', 'coverage', 'mean_se'):
        assert key in result, f"Missing key '{key}' in compute_parameter_metrics output"


def test_mean_se_correct():
    """mean_se should be the arithmetic mean of the provided SEs."""
    from src.metrics import compute_parameter_metrics
    ses = [np.array([0.1 + 0.01 * i]) for i in range(10)]
    estimates = [np.array([1.0]) for _ in range(10)]
    result = compute_parameter_metrics(estimates, np.array([1.0]), ses)
    expected_mean_se = np.mean([0.1 + 0.01 * i for i in range(10)])
    np.testing.assert_allclose(result['mean_se'], [expected_mean_se], atol=1e-10)


# ---------------------------------------------------------------------------
# compute_forecast_metrics
# ---------------------------------------------------------------------------

def test_forecast_metrics_perfect_forecast():
    """MAFE and MSFE must be non-negative; perfect point forecast → both ≈ 0."""
    from src.metrics import compute_forecast_metrics
    T = 300
    # All observations in regime 0, mu_hat[0] = true mean
    y = np.full(T, 1.5)
    pred_probs = np.column_stack([np.ones(T), np.zeros(T)])  # always regime 0
    mu_hat = np.array([1.5, 5.0])
    result = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in=10, end_trim=0)
    assert result['MSFE'] >= 0
    assert result['MAFE'] >= 0
    np.testing.assert_allclose(result['MSFE'], 0.0, atol=1e-10)
    np.testing.assert_allclose(result['MAFE'], 0.0, atol=1e-10)


def test_forecast_metrics_returns_expected_keys():
    from src.metrics import compute_forecast_metrics
    y = np.random.randn(200)
    pred_probs = np.column_stack([np.full(200, 0.6), np.full(200, 0.4)])
    mu_hat = np.array([-1.0, 1.0])
    result = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in=10, end_trim=0)
    assert 'MAFE' in result
    assert 'MSFE' in result


def test_forecast_metrics_burn_in_slice():
    """burn_in should skip the first burn_in observations from evaluation."""
    from src.metrics import compute_forecast_metrics
    T = 100
    y = np.zeros(T)
    pred_probs = np.column_stack([np.ones(T), np.zeros(T)])
    mu_hat = np.array([0.0, 1.0])
    # With burn_in=0: MAFE=0 (perfect), with burn_in=T-1: only last obs counted
    r0 = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in=0, end_trim=0)
    r1 = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in=99, end_trim=0)
    np.testing.assert_allclose(r0['MAFE'], 0.0, atol=1e-10)
    np.testing.assert_allclose(r1['MAFE'], 0.0, atol=1e-10)


def test_forecast_msfe_geq_mafe_squared():
    """MSFE >= MAFE^2 (by Jensen's inequality for convex functions)."""
    from src.metrics import compute_forecast_metrics
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 500)
    pred_probs = np.column_stack([rng.uniform(0, 1, 500)] * 2)
    pred_probs /= pred_probs.sum(axis=1, keepdims=True)
    mu_hat = np.array([-0.5, 0.5])
    r = compute_forecast_metrics(y, pred_probs, mu_hat, burn_in=50, end_trim=0)
    assert r['MSFE'] >= r['MAFE'] ** 2 - 1e-10


# ---------------------------------------------------------------------------
# compute_filtered_prob_errors
# ---------------------------------------------------------------------------

def test_filtered_prob_errors_zero_for_identical():
    """MSE and MAE should be exactly 0 when filtered probs are identical."""
    from src.metrics import compute_filtered_prob_errors
    probs = np.random.default_rng(5).dirichlet(alpha=[1, 2], size=500)
    result = compute_filtered_prob_errors(probs, probs, burn_in=100, end_trim=10)
    assert result['MSE'] < 1e-10
    assert result['MAE'] < 1e-10


def test_filtered_prob_errors_returns_expected_keys():
    from src.metrics import compute_filtered_prob_errors
    T = 200
    hat  = np.random.dirichlet([1, 1], size=T)
    true = np.random.dirichlet([1, 1], size=T)
    result = compute_filtered_prob_errors(hat, true, burn_in=50, end_trim=0)
    assert 'MSE' in result and 'MAE' in result


def test_filtered_prob_errors_non_negative():
    from src.metrics import compute_filtered_prob_errors
    T, K = 300, 2
    hat  = np.random.dirichlet([1] * K, size=T)
    true = np.random.dirichlet([1] * K, size=T)
    result = compute_filtered_prob_errors(hat, true, burn_in=50, end_trim=10)
    assert result['MSE'] >= 0
    assert result['MAE'] >= 0


def test_filtered_prob_errors_mse_leq_1():
    """MSE must be ≤ 1 since probabilities are in [0,1]."""
    from src.metrics import compute_filtered_prob_errors
    T, K = 400, 3
    hat  = np.random.dirichlet([1] * K, size=T)
    true = np.random.dirichlet([1] * K, size=T)
    result = compute_filtered_prob_errors(hat, true, burn_in=0, end_trim=0)
    assert result['MSE'] <= 1.0 + 1e-10


def test_filtered_prob_errors_end_trim():
    """end_trim>0 should exclude the last observations from the metric."""
    from src.metrics import compute_filtered_prob_errors
    T, K = 200, 2
    hat  = np.zeros((T, K)); hat[:, 0] = 1.0
    true = np.zeros((T, K)); true[:, 0] = 1.0
    # Introduce error in last 50 obs
    hat[-50:, 0] = 0.0; hat[-50:, 1] = 1.0

    r_trim = compute_filtered_prob_errors(hat, true, burn_in=10, end_trim=50)
    r_no   = compute_filtered_prob_errors(hat, true, burn_in=10, end_trim=0)
    assert r_trim['MSE'] < r_no['MSE'], 'Trimming erroneous tail should reduce MSE'
