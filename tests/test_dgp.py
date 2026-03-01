"""Tests for data generating processes."""

import numpy as np
import pytest

from draci.dgp import DGPData, AR1DGP, get_dgp


class TestAR1DGP:
    """Tests for AR1DGP class."""

    def test_output_shapes(self):
        """All outputs should have correct dimensions."""
        dgp = AR1DGP(rho=0.5)
        T = 100
        rng = np.random.default_rng(42)

        data = dgp.generate(T, rng)

        assert isinstance(data, DGPData)
        assert data.X.shape == (T,)
        assert data.W.shape == (T,)
        assert data.Y.shape == (T,)
        assert data.e_true.shape == (T,)
        assert data.mu0_true.shape == (T,)
        assert data.mu1_true.shape == (T,)
        assert data.tau_true.shape == (T,)

    def test_treatment_binary(self):
        """Treatment W should be binary (0 or 1)."""
        dgp = AR1DGP(rho=0.5)
        rng = np.random.default_rng(42)

        data = dgp.generate(500, rng)

        assert set(np.unique(data.W)).issubset({0.0, 1.0})

    def test_treatment_rate_approximately_correct(self):
        """Treatment rate should be ~50% (propensity is U-shaped in X)."""
        dgp = AR1DGP(rho=0.0)
        rng = np.random.default_rng(42)

        # Large sample for stable estimate
        data = dgp.generate(5000, rng)
        treatment_rate = data.W.mean()

        # Propensity expit(0.3X + 0.8X^2 - 0.5) averages ~50% for X~N(0,1)
        assert 0.4 < treatment_rate < 0.6

    def test_propensity_bounds(self):
        """Propensity scores should be clipped to [0.05, 0.95]."""
        dgp = AR1DGP(rho=0.9)
        rng = np.random.default_rng(42)

        data = dgp.generate(1000, rng)

        assert np.all(data.e_true >= 0.05)
        assert np.all(data.e_true <= 0.95)

    def test_ar1_stationarity(self):
        """Variance should stabilize for |rho| < 1."""
        rng = np.random.default_rng(42)
        dgp = AR1DGP(rho=0.8)

        data = dgp.generate(2000, rng)

        # Compare variance in first half vs second half
        var_first = np.var(data.X[:1000])
        var_second = np.var(data.X[1000:])

        # Should be similar (ratio close to 1)
        ratio = var_first / var_second
        assert 0.5 < ratio < 2.0

    def test_iid_case(self):
        """When rho=0, X should be approximately IID N(0,1)."""
        dgp = AR1DGP(rho=0.0)
        rng = np.random.default_rng(42)

        data = dgp.generate(5000, rng)

        # Should have mean ~0 and variance ~1
        assert abs(np.mean(data.X)) < 0.1
        assert abs(np.var(data.X) - 1.0) < 0.2

    def test_cate_matches_formula(self):
        """tau_true should match the formula tau(x) = 0.5 + 0.3x + sin(2x)."""
        dgp = AR1DGP(rho=0.5)
        rng = np.random.default_rng(42)

        data = dgp.generate(100, rng)

        expected_tau = 0.5 + 0.3 * data.X + 1.0 * np.sin(2 * data.X)

        np.testing.assert_array_almost_equal(data.tau_true, expected_tau)

    def test_mu1_equals_mu0_plus_tau(self):
        """mu1_true should equal mu0_true + tau_true."""
        dgp = AR1DGP(rho=0.5)
        rng = np.random.default_rng(42)

        data = dgp.generate(100, rng)

        np.testing.assert_array_almost_equal(
            data.mu1_true, data.mu0_true + data.tau_true
        )

    def test_invalid_rho_raises(self):
        """rho >= 1 should raise ValueError."""
        with pytest.raises(ValueError):
            AR1DGP(rho=1.0)

        with pytest.raises(ValueError):
            AR1DGP(rho=1.5)

        with pytest.raises(ValueError):
            AR1DGP(rho=-0.1)

    def test_reproducibility(self):
        """Same seed should produce same data."""
        dgp = AR1DGP(rho=0.5)

        rng1 = np.random.default_rng(42)
        data1 = dgp.generate(100, rng1)

        rng2 = np.random.default_rng(42)
        data2 = dgp.generate(100, rng2)

        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.W, data2.W)
        np.testing.assert_array_equal(data1.Y, data2.Y)


class TestDGPRegistry:
    """Tests for DGP registry."""

    def test_get_ar1_dgp(self):
        """Should retrieve AR1DGP by name."""
        dgp = get_dgp("ar1", rho=0.5)

        assert isinstance(dgp, AR1DGP)
        assert dgp.rho == 0.5

    def test_unknown_dgp_raises(self):
        """Unknown DGP name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown DGP"):
            get_dgp("unknown_dgp")
