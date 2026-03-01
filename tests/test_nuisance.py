"""Tests for nuisance estimators."""

import numpy as np
import pytest

from draci.nuisance import (
    NuisanceFunctions, LinearNuisance, MLNuisance, XGBoostNuisance,
    get_nuisance_estimator, fit_nuisances,
)


class TestLinearNuisance:
    """Tests for LinearNuisance estimator."""

    def test_returns_nuisance_functions(self):
        """fit() should return NuisanceFunctions with tau_hat."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        assert isinstance(funcs, NuisanceFunctions)
        assert callable(funcs.e_hat)
        assert callable(funcs.mu0_hat)
        assert callable(funcs.mu1_hat)
        assert callable(funcs.tau_hat)

    def test_predictions_correct_shape(self):
        """Predictions should match input shape."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)
        assert funcs.mu1_hat(X_test).shape == (50,)
        assert funcs.tau_hat(X_test).shape == (50,)

    def test_tau_hat_equals_mu1_minus_mu0(self):
        """tau_hat should equal mu1_hat - mu0_hat for linear estimator."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)
        np.testing.assert_allclose(
            funcs.tau_hat(X_test),
            funcs.mu1_hat(X_test) - funcs.mu0_hat(X_test),
            rtol=1e-10,
        )

    def test_propensity_bounds(self):
        """Propensity estimates should be clipped to [0.05, 0.95]."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        # Test on extreme X values
        X_extreme = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        e_pred = funcs.e_hat(X_extreme)

        assert np.all(e_pred >= 0.05)
        assert np.all(e_pred <= 0.95)

    def test_recovers_linear_relationship(self):
        """Should approximately recover linear outcome model."""
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.normal(0, 1, n)
        W = rng.binomial(1, 0.5, n).astype(float)

        # True linear model
        mu0_true = 1.0 + 0.5 * X
        mu1_true = 2.0 + 0.5 * X
        Y = np.where(W == 1, mu1_true, mu0_true) + rng.normal(0, 0.1, n)

        estimator = LinearNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = np.linspace(-2, 2, 10)
        mu0_pred = funcs.mu0_hat(X_test)
        mu1_pred = funcs.mu1_hat(X_test)

        mu0_expected = 1.0 + 0.5 * X_test
        mu1_expected = 2.0 + 0.5 * X_test

        # Should be close (within 0.2)
        assert np.max(np.abs(mu0_pred - mu0_expected)) < 0.2
        assert np.max(np.abs(mu1_pred - mu1_expected)) < 0.2


class TestMLNuisance:
    """Tests for MLNuisance estimator."""

    def test_returns_nuisance_functions(self):
        """fit() should return NuisanceFunctions with tau_hat."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = MLNuisance()
        funcs = estimator.fit(X, W, Y)

        assert isinstance(funcs, NuisanceFunctions)
        assert callable(funcs.e_hat)
        assert callable(funcs.mu0_hat)
        assert callable(funcs.mu1_hat)
        assert callable(funcs.tau_hat)

    def test_predictions_correct_shape(self):
        """Predictions should match input shape."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        estimator = MLNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 50)

        assert funcs.e_hat(X_test).shape == (50,)
        assert funcs.mu0_hat(X_test).shape == (50,)
        assert funcs.mu1_hat(X_test).shape == (50,)
        assert funcs.tau_hat(X_test).shape == (50,)

    def test_propensity_bounds(self):
        """Propensity estimates should be clipped to [0.05, 0.95]."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = rng.normal(0, 1, 200)

        estimator = MLNuisance()
        funcs = estimator.fit(X, W, Y)

        X_test = rng.normal(0, 1, 100)
        e_pred = funcs.e_hat(X_test)

        assert np.all(e_pred >= 0.05)
        assert np.all(e_pred <= 0.95)


class TestNuisanceRegistry:
    """Tests for nuisance registry."""

    def test_get_linear(self):
        """Should retrieve LinearNuisance by name."""
        estimator = get_nuisance_estimator("linear")
        assert isinstance(estimator, LinearNuisance)

    def test_get_ml(self):
        """Should retrieve MLNuisance by name."""
        estimator = get_nuisance_estimator("ml")
        assert isinstance(estimator, MLNuisance)

    def test_get_xgboost(self):
        """Should retrieve XGBoostNuisance by name."""
        estimator = get_nuisance_estimator("xgboost")
        assert isinstance(estimator, XGBoostNuisance)

    def test_unknown_raises(self):
        """Unknown estimator name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown nuisance"):
            get_nuisance_estimator("unknown")


class TestFitNuisances:
    """Tests for unified fit_nuisances dispatcher."""

    def test_linear_returns_nuisance_functions(self):
        """fit_nuisances(method='linear') returns NuisanceFunctions with tau_hat."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        funcs = fit_nuisances(X, W, Y, method='linear')
        assert isinstance(funcs, NuisanceFunctions)
        assert callable(funcs.tau_hat)

    def test_tuple_unpacking_backward_compat(self):
        """fit_nuisances result supports 4-element tuple unpacking."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        W = rng.binomial(1, 0.5, 200).astype(float)
        Y = 1 + 0.5 * X + W * 0.3 + rng.normal(0, 0.1, 200)

        e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(
            X, W, Y, method='linear',
        )
        X_test = rng.normal(0, 1, 10)
        assert e_hat_fn(X_test).shape == (10,)
        assert tau_hat_fn(X_test).shape == (10,)
