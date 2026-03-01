"""
Data generating process for DR-ACI simulations (Section 5).

Two DGP variants:
  1. **Production 5D** (generate_data): 5D AR(1) covariates with GARCH(1,1) errors.
  2. **Test 1D** (AR1DGP class): 1D AR(1) with simple propensity/CATE for unit tests.

Production DGP (5D):
  - Covariates: X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t, eps ~ N(0, I_5)
  - Propensity: e(x) = expit(0.5 + 0.8*x1 - 0.3*x1^2 + 0.4*x2)
  - CATE: tau(x) = sin(2*pi*x1) + 0.5*x2
  - Outcome: mu_w(x) = 2 + x1 + 0.5*x2^2 + w*tau(x)
  - Error: u_t = 0.5*u_{t-1} + sigma_t*eta_t (AR(1) + GARCH(1,1))
  - Volatility: sigma_t^2 = 0.1 + 0.3*u_{t-1}^2 + 0.5*sigma_{t-1}^2

Test DGP (1D):
  - Covariates: X_t = rho * X_{t-1} + sqrt(1-rho^2) * eps_t, eps ~ N(0, 1)
  - Propensity: e(x) = expit(0.3*x + 0.8*x^2 - 0.5), clipped to [0.05, 0.95]
  - CATE: tau(x) = 0.5 + 0.3*x + sin(2*x)
  - Outcome: mu0(x) = 1.0 + 0.5*x, mu1 = mu0 + tau
"""

import numpy as np
from typing import NamedTuple

# Mathematical constants of the DGP (not simulation configuration)
DGP_DIM = 5
GARCH_OMEGA = 0.1
GARCH_ALPHA = 0.3
GARCH_BETA = 0.5
AR_ERROR_COEF = 0.5


# =========================================================================
# Test-friendly 1D DGP (class-based API)
# =========================================================================

class DGPData(NamedTuple):
    """Output container for the 1D test DGP."""
    X: np.ndarray        # (T,) covariates
    W: np.ndarray        # (T,) binary treatment
    Y: np.ndarray        # (T,) observed outcomes
    e_true: np.ndarray   # (T,) true propensity scores
    mu0_true: np.ndarray # (T,) true E[Y(0)|X]
    mu1_true: np.ndarray # (T,) true E[Y(1)|X]
    tau_true: np.ndarray # (T,) true CATE


class AR1DGP:
    """1D AR(1) DGP for unit testing.

    Parameters
    ----------
    rho : float in [0, 1)
        AR(1) coefficient.
    """

    def __init__(self, rho: float):
        if rho < 0 or rho >= 1:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        self.rho = rho

    def generate(self, T: int, rng: np.random.Generator) -> DGPData:
        """Generate one trial of (X, W, Y) with ground-truth nuisances."""
        # AR(1) covariates (1D, stationary variance = 1)
        innovation_sd = np.sqrt(max(1.0 - self.rho ** 2, 1e-8))
        X = np.zeros(T)
        X[0] = rng.normal(0, 1)
        for t in range(1, T):
            X[t] = self.rho * X[t - 1] + innovation_sd * rng.normal()

        # Propensity: e(x) = expit(0.3x + 0.8x^2 - 0.5)
        logit = 0.3 * X + 0.8 * X ** 2 - 0.5
        e_true = np.clip(_expit(logit), 0.05, 0.95)

        # Treatment
        W = rng.binomial(1, e_true).astype(float)

        # CATE: tau(x) = 0.5 + 0.3x + sin(2x)
        tau_true = 0.5 + 0.3 * X + 1.0 * np.sin(2 * X)

        # Outcome means
        mu0_true = 1.0 + 0.5 * X
        mu1_true = mu0_true + tau_true

        # Observed outcome with Gaussian noise
        noise = rng.normal(0, 0.5, T)
        Y = np.where(W == 1, mu1_true + noise, mu0_true + noise)

        return DGPData(
            X=X, W=W, Y=Y, e_true=e_true,
            mu0_true=mu0_true, mu1_true=mu1_true, tau_true=tau_true,
        )


def get_dgp(name: str, **kwargs):
    """Factory for DGP instances."""
    if name == "ar1":
        return AR1DGP(**kwargs)
    raise ValueError(f"Unknown DGP: {name}")


def _expit(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def generate_covariates(T: int, rho: float, rng: np.random.Generator,
                        dim: int = DGP_DIM) -> np.ndarray:
    """Generate 5D AR(1) covariates with correlation rho.

    X_t = rho * X_{t-1} + sqrt(1 - rho^2) * eps_t
    where eps_t ~ N(0, I_dim).

    Returns array of shape (T, dim).
    """
    innovation_sd = np.sqrt(max(1.0 - rho**2, 1e-8))
    X = np.zeros((T, dim))
    X[0] = rng.normal(0, 1, dim)
    for t in range(1, T):
        X[t] = rho * X[t - 1] + innovation_sd * rng.normal(0, 1, dim)
    return X


def propensity_true(X: np.ndarray) -> np.ndarray:
    """True propensity score: e(x) = expit(0.5 + 0.8*x1 - 0.3*x1^2 + 0.4*x2).

    Nonlinear in x1 (quadratic confounding), uses x2 for extra variation.
    Clipped to [0.05, 0.95] for overlap (Assumption A3).
    """
    logit = 0.5 + 0.8 * X[:, 0] - 0.3 * X[:, 0]**2 + 0.4 * X[:, 1]
    return np.clip(_expit(logit), 0.05, 0.95)


def cate_true(X: np.ndarray) -> np.ndarray:
    """True CATE: tau(x) = sin(2*pi*x1) + 0.5*x2.

    Strongly nonlinear in x1 via sine; linear in x2.
    """
    return np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1]


def outcome_mean(X: np.ndarray, w: int) -> np.ndarray:
    """Conditional mean: mu_w(x) = 2 + x1 + 0.5*x2^2 + w*tau(x).

    mu0 is nonlinear only through x2^2; mu1 adds the nonlinear CATE.
    """
    base = 2.0 + X[:, 0] + 0.5 * X[:, 1]**2
    if w == 1:
        return base + cate_true(X)
    return base


def generate_garch_errors(T: int, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) errors with AR(1) in level.

    u_t = phi * u_{t-1} + sigma_t * eta_t
    sigma_t^2 = omega + alpha_g * u_{t-1}^2 + beta_g * sigma_{t-1}^2
    eta_t ~ N(0, 1)

    Parameters: omega=0.1, alpha_g=0.3, beta_g=0.5, phi=0.5.
    """
    phi = AR_ERROR_COEF
    omega = GARCH_OMEGA
    alpha_g = GARCH_ALPHA
    beta_g = GARCH_BETA

    u = np.zeros(T)
    sigma2 = np.zeros(T)
    eta = rng.normal(0, 1, T)

    # Initialize at unconditional variance: sigma^2_unc = omega / (1 - alpha - beta)
    sigma2_unc = omega / max(1.0 - alpha_g - beta_g, 0.01)
    sigma2[0] = sigma2_unc
    u[0] = np.sqrt(sigma2[0]) * eta[0]

    for t in range(1, T):
        sigma2[t] = omega + alpha_g * u[t - 1]**2 + beta_g * sigma2[t - 1]
        u[t] = phi * u[t - 1] + np.sqrt(sigma2[t]) * eta[t]

    return u


def generate_data(T: int, rho: float, rng: np.random.Generator) -> dict:
    """Generate complete DGP for one simulation trial.

    Returns dict with keys:
        X        : (T, 5) covariates
        W        : (T,)   treatment indicators
        Y        : (T,)   observed outcomes
        e_true   : (T,)   true propensity scores
        mu0_true : (T,)   true E[Y(0)|X]
        mu1_true : (T,)   true E[Y(1)|X]
        tau_true : (T,)   true CATE
    """
    X = generate_covariates(T, rho, rng)
    e_true = propensity_true(X)
    tau = cate_true(X)
    mu0 = outcome_mean(X, 0)
    mu1 = outcome_mean(X, 1)

    W = rng.binomial(1, e_true)
    u = generate_garch_errors(T, rng)

    Y0 = mu0 + u
    Y1 = mu1 + u
    Y = np.where(W == 1, Y1, Y0)

    return {
        'X': X,
        'W': W,
        'Y': Y,
        'e_true': e_true,
        'mu0_true': mu0,
        'mu1_true': mu1,
        'tau_true': tau,
    }
