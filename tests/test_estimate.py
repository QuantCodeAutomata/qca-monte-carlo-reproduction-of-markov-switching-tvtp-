"""Tests for src/estimate.py — pack/unpack, MLE, and label switching."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# pack_params / unpack_params roundtrip
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip_constant():
    """pack then unpack should recover original parameters for Constant K=2."""
    from src.estimate import pack_params, unpack_params
    mu = np.array([-1.0, 1.0])
    log_sigma2 = np.array([np.log(0.5)])   # single element: common variance
    omega = np.array([1.386, 2.197])
    # pack_params takes (mu, log_sigma2, omega, A=None, B=None, model='Constant')
    theta = pack_params(mu, log_sigma2, omega, model='Constant')
    params = unpack_params(theta, K=2, parameterization='diagonal',
                           model='Constant', common_variance=True)
    np.testing.assert_allclose(params['mu'], mu, atol=1e-10)
    np.testing.assert_allclose(params['sigma2'], np.array([0.5, 0.5]), atol=1e-10)
    np.testing.assert_allclose(params['omega'], omega, atol=1e-10)


def test_pack_unpack_roundtrip_model_i():
    """pack/unpack roundtrip for ModelI (adds A parameter)."""
    from src.estimate import pack_params, unpack_params
    mu = np.array([-1.0, 1.0])
    log_sigma2 = np.array([np.log(0.5)])
    omega = np.array([1.386, 2.197])
    A = np.array([0.2, -0.15])
    theta = pack_params(mu, log_sigma2, omega, A=A, model='ModelI')
    params = unpack_params(theta, K=2, parameterization='diagonal',
                           model='ModelI', common_variance=True)
    np.testing.assert_allclose(params['mu'], mu, atol=1e-10)
    np.testing.assert_allclose(params['A'], A, atol=1e-10)
    np.testing.assert_allclose(params['omega'], omega, atol=1e-10)


def test_pack_unpack_roundtrip_model_iii():
    """pack/unpack roundtrip for ModelIII (adds A and B)."""
    from src.estimate import pack_params, unpack_params
    mu = np.array([-1.0, 1.0])
    log_sigma2 = np.array([np.log(0.5)])
    omega = np.array([1.386, 2.197])
    A = np.array([0.1, -0.08])
    B = np.array([0.9, 0.85])
    theta = pack_params(mu, log_sigma2, omega, A=A, B=B, model='ModelIII')
    params = unpack_params(theta, K=2, parameterization='diagonal',
                           model='ModelIII', common_variance=True)
    np.testing.assert_allclose(params['mu'], mu, atol=1e-10)
    np.testing.assert_allclose(params['A'], A, atol=1e-10)
    np.testing.assert_allclose(params['B'], B, atol=1e-10)


def test_pack_unpack_k3_off_diagonal():
    """pack/unpack roundtrip for K=3 off_diagonal Constant model."""
    from src.estimate import pack_params, unpack_params
    K = 3
    mu = np.array([-2.0, 0.0, 2.0])
    sigma2_vals = np.array([0.3, 0.5, 0.8])
    log_sigma2 = np.log(sigma2_vals)        # 3 elements: regime-specific variance
    omega = np.full(6, -2.5)               # K*(K-1)=6 params
    theta = pack_params(mu, log_sigma2, omega, model='Constant')
    params = unpack_params(theta, K=K, parameterization='off_diagonal',
                           model='Constant', common_variance=False)
    np.testing.assert_allclose(params['mu'], mu, atol=1e-10)
    np.testing.assert_allclose(params['sigma2'], sigma2_vals, atol=1e-10)
    np.testing.assert_allclose(params['omega'], omega, atol=1e-10)


def test_unpack_sigma2_exponentiation():
    """unpack_params should apply exp() to log_sigma2 to recover sigma2."""
    from src.estimate import pack_params, unpack_params
    mu = np.array([-1.0, 1.0])
    true_sigma2 = 0.7
    log_sigma2 = np.array([np.log(true_sigma2)])
    omega = np.array([1.0, 1.0])
    theta = pack_params(mu, log_sigma2, omega, model='Constant')
    params = unpack_params(theta, K=2, parameterization='diagonal',
                           model='Constant', common_variance=True)
    np.testing.assert_allclose(params['sigma2'], np.full(2, true_sigma2), atol=1e-10)


# ---------------------------------------------------------------------------
# neg_loglik
# ---------------------------------------------------------------------------

def test_neg_loglik_finite_for_good_params():
    """neg_loglik should be finite for a well-specified model."""
    from src.estimate import neg_loglik
    np.random.seed(42)
    y = np.random.randn(300)
    mu = np.array([-1.0, 1.0])
    log_sigma2 = np.array([np.log(0.5)])
    omega = np.array([1.386, 2.197])
    from src.estimate import pack_params
    theta = pack_params(mu, log_sigma2, omega, model='Constant')
    nll = neg_loglik(theta, y, K=2, parameterization='diagonal',
                     model='Constant', common_variance=True,
                     burn_in=10, end_trim=0)
    assert np.isfinite(nll), f'neg_loglik={nll} should be finite'
    assert nll > 0, 'neg_loglik should be positive (= −loglik)'


def test_neg_loglik_returns_scalar():
    """neg_loglik must return a Python scalar, not an array."""
    from src.estimate import neg_loglik, pack_params
    y = np.random.randn(100)
    theta = pack_params(np.array([-1.0, 1.0]),
                        np.array([np.log(0.5)]),
                        np.array([1.0, 1.0]),
                        model='Constant')
    nll = neg_loglik(theta, y, K=2, parameterization='diagonal',
                     model='Constant', common_variance=True,
                     burn_in=10, end_trim=0)
    assert np.ndim(nll) == 0, 'neg_loglik should return a scalar'


# ---------------------------------------------------------------------------
# estimate_model — convergence
# ---------------------------------------------------------------------------

def test_estimation_constant_k2_converges():
    """Estimation of K=2 Constant model should converge on DGP 1 data."""
    from src.estimate import estimate_model
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[1]   # K=2, mu=[-1,1], sigma2=0.5
    rng = np.random.default_rng(42)
    y, _, _ = simulate_dgp(dgp, T=500, rng=rng)
    result = estimate_model(
        y, K=2, parameterization='diagonal',
        model='Constant', common_variance=True,
        n_starts=3, burn_in=100, end_trim=10,
        rng=np.random.default_rng(0),
    )
    assert result['converged'], 'Should converge on DGP1 data'
    assert np.isfinite(result['loglik']), 'loglik must be finite'
    mu_hat = result['params']['mu']
    # After label switching correction, one estimate < 0 and one > 0
    assert np.any(mu_hat < 0) and np.any(mu_hat > 0), (
        f'mu_hat={mu_hat}: expected one negative and one positive estimate'
    )


def test_estimate_model_result_keys():
    """estimate_model result dict must contain required keys."""
    from src.estimate import estimate_model
    y = np.random.randn(200)
    result = estimate_model(
        y, K=2, parameterization='diagonal',
        model='Constant', common_variance=True,
        n_starts=2, burn_in=10, end_trim=0,
        rng=np.random.default_rng(1),
    )
    for key in ('converged', 'loglik', 'params', 'se',
                'filtered_probs', 'pred_probs'):
        assert key in result, f"Missing key '{key}' in estimate_model result"


def test_estimate_model_filtered_probs_shape():
    """Filtered probs returned by estimate_model must have shape (T, K)."""
    from src.estimate import estimate_model
    T = 200
    y = np.random.randn(T)
    result = estimate_model(
        y, K=2, parameterization='diagonal',
        model='Constant', common_variance=True,
        n_starts=2, burn_in=10, end_trim=0,
        rng=np.random.default_rng(2),
    )
    if result['converged']:
        fp = result['filtered_probs']
        assert fp.shape == (T, 2), f'Expected ({T}, 2), got {fp.shape}'
        np.testing.assert_allclose(fp.sum(axis=1), np.ones(T), atol=1e-8)


# ---------------------------------------------------------------------------
# correct_label_switching
# ---------------------------------------------------------------------------

def test_label_switching_correction_swaps_correctly():
    """correct_label_switching should un-swap regimes when mu is reversed."""
    from src.estimate import correct_label_switching
    estimated_mu = np.array([1.0, -1.0])   # switched vs true [-1, 1]
    true_mu = np.array([-1.0, 1.0])
    params = {
        'mu': estimated_mu,
        'sigma2': np.array([0.5, 0.5]),
        'omega': np.array([2.0, 1.0]),
    }
    rng = np.random.default_rng(7)
    filtered_probs = rng.dirichlet([1, 1], size=100)
    pred_probs = rng.dirichlet([1, 1], size=100)

    corrected_params, corrected_filt, _ = correct_label_switching(
        estimated_mu, true_mu, params, filtered_probs, pred_probs,
        K=2, parameterization='diagonal',
    )
    np.testing.assert_allclose(corrected_params['mu'], [-1.0, 1.0], atol=1e-10)


def test_label_switching_no_swap_when_aligned():
    """When mu is already aligned, correction should leave params unchanged."""
    from src.estimate import correct_label_switching
    mu = np.array([-1.0, 1.0])
    true_mu = np.array([-1.0, 1.0])
    params = {'mu': mu, 'sigma2': np.array([0.5, 0.5]), 'omega': np.array([1.0, 2.0])}
    rng = np.random.default_rng(99)
    fp = rng.dirichlet([1, 1], size=50)
    pp = rng.dirichlet([1, 1], size=50)
    corrected, _, _ = correct_label_switching(mu, true_mu, params, fp, pp,
                                              K=2, parameterization='diagonal')
    np.testing.assert_allclose(corrected['mu'], [-1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(corrected['sigma2'], [0.5, 0.5], atol=1e-10)


def test_label_switching_permutes_filtered_probs():
    """Label switching should consistently permute filtered_probs columns."""
    from src.estimate import correct_label_switching
    estimated_mu = np.array([2.0, -2.0])   # regime 0↔1 swapped
    true_mu = np.array([-2.0, 2.0])
    params = {'mu': estimated_mu, 'sigma2': np.array([1.0, 1.0]),
              'omega': np.array([1.0, 1.0])}
    T = 100
    fp = np.zeros((T, 2))
    fp[:, 0] = 0.9; fp[:, 1] = 0.1   # column 0 = estimated regime 0 (true regime 1)
    pp = fp.copy()

    corrected, corrected_fp, _ = correct_label_switching(
        estimated_mu, true_mu, params, fp, pp,
        K=2, parameterization='diagonal',
    )
    # After correction, column 0 should be true regime 0 (was column 1 = 0.1)
    np.testing.assert_allclose(corrected_fp[:, 0], np.full(T, 0.1), atol=1e-10)
    np.testing.assert_allclose(corrected_fp[:, 1], np.full(T, 0.9), atol=1e-10)


def test_label_switching_k3():
    """correct_label_switching works for K=3 regimes."""
    from src.estimate import correct_label_switching
    # True order: [-2, 0, 2]; estimated: [2, -2, 0] (permuted)
    estimated_mu = np.array([2.0, -2.0, 0.0])
    true_mu = np.array([-2.0, 0.0, 2.0])
    params = {
        'mu': estimated_mu,
        'sigma2': np.array([0.8, 0.3, 0.5]),
    }
    rng = np.random.default_rng(33)
    fp = rng.dirichlet([1, 1, 1], size=100)
    pp = rng.dirichlet([1, 1, 1], size=100)
    corrected, _, _ = correct_label_switching(
        estimated_mu, true_mu, params, fp, pp,
        K=3, parameterization='off_diagonal',
    )
    np.testing.assert_allclose(corrected['mu'], [-2.0, 0.0, 2.0], atol=1e-10)
