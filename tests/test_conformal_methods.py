"""Tests for conformal prediction methods."""

import numpy as np
import pytest

from draci.conformal import (
    dr_score, naive_score, split_conformal, aci, dr_aci, ConformalResult
)


class TestDRScore:
    """Tests for dr_score function."""

    def test_output_shape(self):
        """DR score output should match input shape."""
        T = 100
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, T)
        e_hat = np.full(T, 0.5)
        mu0_hat = np.zeros(T)
        mu1_hat = np.ones(T)
        tau_hat = np.ones(T)

        scores = dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat)

        assert scores.shape == (T,)
        assert np.all(scores >= 0)  # Absolute values are non-negative

    def test_perfect_nuisance_small_scores(self):
        """With perfect nuisances, DR scores should be small (just noise)."""
        T = 500
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, T)
        e_true = np.full(T, 0.5)
        tau_true = 0.5 + 0.3 * X
        mu0_true = 1.0 + 0.5 * X
        mu1_true = mu0_true + tau_true

        W = rng.binomial(1, e_true).astype(float)
        sigma_eta = 0.1
        eta = rng.normal(0, sigma_eta, T)
        Y = np.where(W == 1, mu1_true + eta, mu0_true + eta)

        # Use true nuisances
        scores = dr_score(Y, W, X, e_true, mu0_true, mu1_true, tau_true)

        # With perfect nuisances, scores should be O(sigma_eta)
        assert np.mean(scores) < 0.5

    def test_scores_nonnegative(self):
        """DR scores are absolute values, hence non-negative."""
        T = 50
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, T)
        e_hat = np.clip(rng.uniform(0.2, 0.8, T), 0.05, 0.95)
        mu0_hat = rng.normal(0, 1, T)
        mu1_hat = rng.normal(1, 1, T)
        tau_hat = rng.normal(0.5, 0.5, T)

        scores = dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat)

        assert np.all(scores >= 0)


class TestNaiveScore:
    """Tests for naive_score function."""

    def test_output_shape(self):
        """Naive score output should match input shape."""
        T = 100
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, T)
        mu0_hat = np.zeros(T)
        mu1_hat = np.ones(T)
        tau_hat = np.ones(T)

        scores = naive_score(Y, W, X, mu0_hat, mu1_hat, tau_hat)

        assert scores.shape == (T,)
        assert np.all(scores >= 0)


class TestSplitConformal:
    """Tests for split_conformal function."""

    def test_returns_conformal_result(self):
        """split_conformal should return ConformalResult."""
        rng = np.random.default_rng(42)
        scores_cal = rng.exponential(1, 100)
        scores_test = rng.exponential(1, 100)
        true_residuals = rng.exponential(1, 100)

        result = split_conformal(scores_cal, scores_test, true_residuals, alpha=0.1)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0
        assert len(result.coverages_t) == len(true_residuals)

    def test_iid_coverage_near_nominal(self):
        """On IID data, split conformal should achieve ~(1-alpha) coverage."""
        rng = np.random.default_rng(42)
        alpha = 0.1
        n_trials = 50
        coverages = []

        for _ in range(n_trials):
            # IID scores from same distribution
            scores_cal = rng.exponential(1, 200)
            scores_test = rng.exponential(1, 200)
            true_residuals = rng.exponential(1, 200)  # Same distribution

            result = split_conformal(scores_cal, scores_test, true_residuals, alpha=alpha)
            coverages.append(result.coverage)

        avg_coverage = np.mean(coverages)
        # Should be within 0.1 of nominal (accounting for finite sample)
        assert abs(avg_coverage - (1 - alpha)) < 0.1


class TestACI:
    """Tests for aci function."""

    def test_returns_conformal_result(self):
        """aci should return ConformalResult."""
        rng = np.random.default_rng(42)
        scores = rng.exponential(1, 200)
        true_residuals = rng.exponential(1, 200)

        result = aci(scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=20)

        assert isinstance(result, ConformalResult)
        assert 0 <= result.coverage <= 1
        assert result.avg_width > 0

    def test_converges_to_nominal(self):
        """ACI should converge toward nominal coverage on stationary data."""
        rng = np.random.default_rng(42)
        T = 1000
        alpha = 0.1

        # IID scores
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        result = aci(scores, true_residuals, alpha=alpha, gamma=0.02, n_warmup=50)

        # Should be reasonably close to nominal
        assert abs(result.coverage - (1 - alpha)) < 0.15


class TestDRACI:
    """Tests for dr_aci function."""

    def test_same_as_aci(self):
        """dr_aci should produce same results as aci (it's a wrapper)."""
        rng = np.random.default_rng(42)
        scores = rng.exponential(1, 200)
        true_residuals = rng.exponential(1, 200)

        result_aci = aci(scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=20)
        result_draci = dr_aci(scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=20)

        assert result_aci.coverage == result_draci.coverage
        assert result_aci.avg_width == result_draci.avg_width


class TestEdgeCases:
    """Edge case tests."""

    def test_small_sample(self):
        """Methods should handle small samples without crashing."""
        rng = np.random.default_rng(42)
        T = 20
        scores = rng.exponential(1, T)
        true_residuals = rng.exponential(1, T)

        # Should not raise
        result = aci(scores, true_residuals, alpha=0.1, gamma=0.01, n_warmup=5)
        assert isinstance(result, ConformalResult)

    def test_extreme_propensities(self):
        """DR score should handle propensities near boundaries."""
        T = 50
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, T)
        W = rng.binomial(1, 0.5, T).astype(float)
        X = rng.normal(0, 1, T)

        # Propensities near boundaries (but clipped)
        e_hat = np.clip(rng.uniform(0.01, 0.99, T), 0.05, 0.95)
        mu0_hat = np.zeros(T)
        mu1_hat = np.ones(T)
        tau_hat = np.ones(T)

        scores = dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(scores))
