"""Tests for src/simulate.py — simulate_dgp output shapes and distributions."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Output shapes and types
# ---------------------------------------------------------------------------

def test_simulate_dgp1_shapes():
    """DGP 1 (K=2, Constant): output shapes and X is None."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgps = get_all_dgps()
    dgp = dgps[1]
    rng = np.random.default_rng(42)
    y, z, X = simulate_dgp(dgp, T=500, rng=rng)
    assert y.shape == (500,)
    assert z.shape == (500,)
    assert X is None
    assert np.all(np.isin(z, [0, 1])), 'Regime indices must be 0 or 1 for K=2'


def test_simulate_dgp3_with_exogenous():
    """DGP 3 (ModelII): exogenous series X should be returned with correct shape."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgps = get_all_dgps()
    dgp = dgps[3]   # ModelII DGP
    rng = np.random.default_rng(42)
    y, z, X = simulate_dgp(dgp, T=500, rng=rng)
    assert y.shape == (500,)
    assert X is not None, 'ModelII DGP must return exogenous X'
    assert X.shape == (500,), f'X shape {X.shape} != (500,)'


def test_simulate_dgp6_k3():
    """DGP 6 (K=3, Constant): regime indices should cover all three regimes."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgps = get_all_dgps()
    dgp = dgps[6]
    rng = np.random.default_rng(42)
    y, z, X = simulate_dgp(dgp, T=500, rng=rng)
    assert y.shape == (500,)
    assert z.shape == (500,)
    assert X is None
    assert np.all(np.isin(z, [0, 1, 2])), 'K=3 regimes must be 0, 1, or 2'
    regime_counts = np.bincount(z, minlength=3)
    assert all(rc > 10 for rc in regime_counts), (
        f'All regimes should appear at T=500; counts={regime_counts}'
    )


def test_simulate_y_is_float():
    """Simulated y must be a float array, z must be integer."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[1]
    rng = np.random.default_rng(0)
    y, z, _ = simulate_dgp(dgp, T=200, rng=rng)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.issubdtype(z.dtype, np.integer)


def test_simulate_reproducible_with_same_seed():
    """Same seed → identical output; different seed → different output."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[1]
    y1, z1, _ = simulate_dgp(dgp, T=300, rng=np.random.default_rng(99))
    y2, z2, _ = simulate_dgp(dgp, T=300, rng=np.random.default_rng(99))
    y3, z3, _ = simulate_dgp(dgp, T=300, rng=np.random.default_rng(777))
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(y1, y3), 'Different seeds should give different data'


# ---------------------------------------------------------------------------
# Statistical distribution checks
# ---------------------------------------------------------------------------

def test_simulate_y_distribution_per_regime():
    """Check that y|z=i is approximately N(mu_i, sigma2_i) for DGP 1."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgps = get_all_dgps()
    dgp = dgps[1]   # K=2, mu=[-1,1], sigma2=0.5 (common)
    rng = np.random.default_rng(0)
    y, z, _ = simulate_dgp(dgp, T=5000, rng=rng)
    for i, (mu_i, sig2_i) in enumerate(zip([-1.0, 1.0], [0.5, 0.5])):
        y_i = y[z == i]
        if len(y_i) > 50:
            assert abs(y_i.mean() - mu_i) < 0.15, (
                f'Regime {i}: mean={y_i.mean():.3f}, expected≈{mu_i}'
            )
            assert abs(y_i.var() - sig2_i) < 0.15, (
                f'Regime {i}: var={y_i.var():.3f}, expected≈{sig2_i}'
            )


def test_simulate_k3_regime_means():
    """K=3 DGP (DGP 6): each regime's y should cluster near the correct mean."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[6]   # mu=[-2, 0, 2]
    rng = np.random.default_rng(0)
    y, z, _ = simulate_dgp(dgp, T=10000, rng=rng)
    for i, mu_i in enumerate([-2.0, 0.0, 2.0]):
        y_i = y[z == i]
        if len(y_i) > 100:
            assert abs(y_i.mean() - mu_i) < 0.3, (
                f'Regime {i}: mean={y_i.mean():.3f}, expected≈{mu_i}'
            )


def test_simulate_all_observations_finite():
    """All simulated y and z values should be finite and valid."""
    from src.simulate import simulate_dgp
    from config.dgp_params import get_all_dgps
    for dgp_id in [1, 2, 3, 4, 5, 6]:
        dgp = get_all_dgps()[dgp_id]
        rng = np.random.default_rng(dgp_id)
        y, z, X = simulate_dgp(dgp, T=200, rng=rng)
        assert np.all(np.isfinite(y)), f'DGP{dgp_id}: non-finite y'
        assert np.all(np.isfinite(z)), f'DGP{dgp_id}: non-finite z'
        if X is not None:
            assert np.all(np.isfinite(X)), f'DGP{dgp_id}: non-finite X'


# ---------------------------------------------------------------------------
# DGP configuration consistency
# ---------------------------------------------------------------------------

def test_dgp_k2_constant_no_a():
    """DGP 1 (Constant) has A=None."""
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[1]
    assert dgp.A is None, 'Constant DGP should have A=None'
    assert dgp.B is None, 'Constant DGP should have B=None'


def test_dgp_gas_has_a_and_b():
    """DGP 4 (ModelIII/GAS) must have both A and B."""
    from config.dgp_params import get_all_dgps
    dgp = get_all_dgps()[4]
    assert dgp.A is not None, 'GAS DGP must have A'
    assert dgp.B is not None, 'GAS DGP must have B'
    assert len(dgp.A) == len(dgp.B), 'A and B must have same length'


def test_get_omega_returns_correct_shape():
    """DGPConfig.get_omega() should return correct number of logit parameters."""
    from config.dgp_params import get_all_dgps
    dgps = get_all_dgps()
    # K=2, diagonal → 2 omega params
    omega2 = dgps[1].get_omega()
    assert omega2.shape == (2,)
    # K=3, off_diagonal → K*(K-1)=6 omega params
    omega6 = dgps[6].get_omega()
    assert omega6.shape == (6,)
