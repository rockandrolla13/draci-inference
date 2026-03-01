"""
Conformal prediction methods for CATE under mixing.

Implements nine methods for the DR-ACI coverage study (Section 5 of the paper):

1. dr_aci        -- Doubly robust adaptive conformal inference (our method)
2. vs_dr_aci     -- Variance-standardized DR-ACI (tighter intervals)
3. split_conformal -- Standard split conformal (ignores dependence)
4. nexcp         -- NexCP: weighted conformal with exponential decay
5. aci           -- ACI with plug-in (non-DR) conformity scores
6. eci           -- Error-quantified conformal inference (smooth feedback)
7. block_cp      -- Block-permutation conformal prediction
8. hac           -- HAC (Newey-West) asymptotic confidence intervals
9. oracle        -- DR-ACI with true nuisance functions

DR conformity score (Assumption A5):
    S_t^{DR} = |psi_t^{DR} - tau_hat(X_t)|
where
    psi_t^{DR} = W_t/e_hat(X_t) * (Y_t - mu1_hat(X_t))
               - (1-W_t)/(1-e_hat(X_t)) * (Y_t - mu0_hat(X_t))
               + mu1_hat(X_t) - mu0_hat(X_t)

IMPORTANT: Coverage is measured as TRUE coverage -- whether tau_true(X_t)
falls in the interval [tau_hat - q_t, tau_hat + q_t] -- NOT self-coverage
of the conformity score.
"""

import numpy as np
from typing import NamedTuple
from scipy import stats


class ConformalResult(NamedTuple):
    """Result from a conformal method."""
    coverage: float          # TRUE coverage: fraction of time tau_true in interval
    avg_width: float         # average prediction interval width (full width, 2*q)
    coverages_t: np.ndarray  # per-time-step TRUE coverage indicators


# =========================================================================
# Score functions
# =========================================================================

def dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat):
    """Doubly robust conformity score |psi^DR - tau_hat|.

    Parameters
    ----------
    Y : array (T,)   observed outcomes
    W : array (T,)   treatment indicators
    X : array (T,d)  covariates (unused, interface consistency)
    e_hat : array (T,)  estimated propensity
    mu0_hat, mu1_hat : array (T,)  estimated outcome means
    tau_hat : array (T,)  CATE point predictions

    Returns
    -------
    scores : array (T,)  |psi^DR - tau_hat|
    """
    psi_dr = (
        W / e_hat * (Y - mu1_hat)
        - (1 - W) / (1 - e_hat) * (Y - mu0_hat)
        + mu1_hat - mu0_hat
    )
    return np.abs(psi_dr - tau_hat)


def dr_pseudo_outcome(Y, W, X, e_hat, mu0_hat, mu1_hat):
    """DR pseudo-outcome psi^DR (signed, before taking absolute value).

    Useful for variance estimation and HAC methods.
    """
    return (
        W / e_hat * (Y - mu1_hat)
        - (1 - W) / (1 - e_hat) * (Y - mu0_hat)
        + mu1_hat - mu0_hat
    )


def naive_score(Y, W, X, mu0_hat, mu1_hat, tau_hat):
    """Non-DR conformity score: plug-in residual without IPW.

    Returns |Y_t - mu_{W_t}(X_t) - tau_hat(X_t) * (something)|
    """
    residual = np.where(W == 1, Y - mu1_hat, -(Y - mu0_hat))
    return np.abs(residual + (mu1_hat - mu0_hat) - tau_hat)


def estimate_if_variance(Y, W, e_hat, mu0_hat, mu1_hat):
    """Influence-function variance estimate sigma^2(X).

    sigma^2(x) = Var(Y(1)|x)/e(x) + Var(Y(0)|x)/(1-e(x))

    Approximated from squared residuals.
    Paper equations (7)-(8).
    """
    res1 = (Y - mu1_hat)**2
    res0 = (Y - mu0_hat)**2
    sigma2_hat = res1 / e_hat + res0 / (1 - e_hat)
    return np.sqrt(np.maximum(sigma2_hat, 1e-6))


def vs_dr_score(Y, W, X, e_hat, mu0_hat, mu1_hat, tau_hat):
    """Variance-standardized DR conformity score.

    S_t = |psi^DR - tau_hat| / sigma_hat(X_t)

    sigma_hat is the influence-function standard deviation estimate;
    see estimate_if_variance().
    """
    psi_dr = dr_pseudo_outcome(Y, W, X, e_hat, mu0_hat, mu1_hat)
    sigma_hat = estimate_if_variance(Y, W, e_hat, mu0_hat, mu1_hat)

    return np.abs(psi_dr - tau_hat) / sigma_hat


# =========================================================================
# Method 1: DR-ACI (our method)
# =========================================================================

def dr_aci(dr_scores_seq, true_residuals, alpha=0.1, gamma=0.005, n_warmup=50):
    """DR-ACI: adaptive conformal with DR conformity scores.

    Uses DR scores from dr_score() with the ACI online update (Algorithm 1).
    """
    return aci(dr_scores_seq, true_residuals, alpha=alpha, gamma=gamma,
               n_warmup=n_warmup)


# =========================================================================
# Method 2: VS-DR-ACI (variance-standardized)
# =========================================================================

def vs_dr_aci(vs_scores_seq, true_residuals_standardized, true_residuals,
              alpha=0.1, gamma=0.005, n_warmup=50):
    """VS-DR-ACI: DR-ACI with variance-standardized scores.

    ACI runs on standardized scores. Coverage is measured on original
    scale by comparing sigma_hat * q_t to true_residuals (unstandardized).

    Parameters
    ----------
    vs_scores_seq : standardized DR scores |psi^DR - tau_hat| / sigma_hat
    true_residuals_standardized : |tau_true - tau_hat| / sigma_hat
    true_residuals : |tau_true - tau_hat| (unstandardized, for reference)
    """
    T = len(vs_scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = vs_scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        # Self-coverage on standardized scores
        self_covered_t = float(vs_scores_seq[t] <= q_hat)
        # True coverage: compare standardized true residual to standardized quantile
        true_covered[t] = float(true_residuals_standardized[t] <= q_hat)
        # Width in standardized units (will be rescaled when reporting)
        widths[t] = 2 * q_hat

        err_t = 1 - self_covered_t
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
    )


# =========================================================================
# Method 3: Split conformal
# =========================================================================

def split_conformal(scores_cal, scores_test, true_residuals_test, alpha=0.1):
    """Standard split conformal: fixed quantile from calibration set.

    No online adaptation. Calibration scores set the quantile.
    """
    n_cal = len(scores_cal)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_hat = np.quantile(scores_cal, min(q_level, 1.0))
    true_covered = (true_residuals_test <= q_hat).astype(float)
    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * q_hat,
        coverages_t=true_covered,
    )


# =========================================================================
# Method 4: NexCP (Barber et al. 2023)
# =========================================================================

def nexcp(scores_seq, true_residuals, alpha=0.1, lam=0.05, n_warmup=50):
    """NexCP: weighted quantile with exponential decay weights.

    At time t, compute weighted quantile of past scores with weights
    w_s = exp(-lam * (t - s)) for s < t.

    Parameters
    ----------
    lam : float
        Exponential decay rate. Larger = more weight on recent scores.
    """
    T = len(scores_seq)
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        # Weights: exponential decay
        lags = np.arange(t, 0, -1, dtype=float)  # t, t-1, ..., 1
        weights = np.exp(-lam * lags)
        cal_scores = scores_seq[:t]

        # Weighted quantile via sorting
        sorted_idx = np.argsort(cal_scores)
        sorted_scores = cal_scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        cum_weights /= cum_weights[-1]  # normalize to [0, 1]

        # Find the quantile level
        q_level = 1 - alpha
        idx = np.searchsorted(cum_weights, q_level)
        idx = min(idx, len(sorted_scores) - 1)
        q_hat = sorted_scores[idx]

        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
    )


# =========================================================================
# Method 5: ACI (non-DR) — Gibbs & Candes 2021
# =========================================================================

def aci(scores_seq, true_residuals, alpha=0.1, gamma=0.005, n_warmup=50):
    """Adaptive conformal inference with online recalibration.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1{S_t > q_hat_t} (self-coverage drives update).

    TRUE coverage measured separately: 1{true_residuals[t] <= q_hat_t}.
    """
    T = len(scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        self_covered_t = float(scores_seq[t] <= q_hat)
        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat

        err_t = 1 - self_covered_t
        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
    )


# =========================================================================
# Method 6: ECI — Error-quantified Conformal Inference (Wu et al. ICLR 2025)
# =========================================================================

def eci(scores_seq, true_residuals, alpha=0.1, gamma=0.005, n_warmup=50,
        temperature=1.0):
    """ECI: ACI with smooth sigmoid feedback instead of binary indicator.

    Replaces err_t = 1{S_t > q_t} with a smooth sigmoid:
        err_t = sigmoid((S_t - q_t) / temperature)

    This reduces oscillation in the quantile update, producing more
    stable interval widths.

    Parameters
    ----------
    temperature : float
        Controls sigmoid sharpness. Lower = closer to hard indicator.
    """
    T = len(scores_seq)
    alpha_t = alpha
    true_covered = np.zeros(T)
    widths = np.zeros(T)

    for t in range(n_warmup, T):
        cal_scores = scores_seq[:t]
        q_hat = np.quantile(cal_scores, min(max(1 - alpha_t, 0), 1))

        # Smooth sigmoid feedback
        diff = (scores_seq[t] - q_hat) / max(temperature, 1e-6)
        # Numerically stable sigmoid
        smooth_err = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))

        true_covered[t] = float(true_residuals[t] <= q_hat)
        widths[t] = 2 * q_hat

        # ACI update with smooth error
        alpha_t = alpha_t + gamma * (alpha - smooth_err)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

    valid = slice(n_warmup, T)
    return ConformalResult(
        coverage=true_covered[valid].mean(),
        avg_width=widths[valid].mean(),
        coverages_t=true_covered[valid],
    )


# =========================================================================
# Method 7: Block CP — Block-permutation conformal (Chernozhukov et al. 2018)
# =========================================================================

def block_cp(scores_seq, true_residuals, alpha=0.1, block_size=10, n_reps=200,
             rng=None):
    """Block-permutation conformal prediction.

    Partition calibration scores into non-overlapping blocks of size b.
    Permute entire blocks (preserving within-block dependence).
    Compute p-value from block-permuted quantiles.

    Parameters
    ----------
    block_size : int
        Size of each block for permutation.
    n_reps : int
        Number of block-permutation replicates.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = len(scores_seq)
    n_cal = T // 2
    cal_scores = scores_seq[:n_cal]
    test_scores = scores_seq[n_cal:]
    test_true_res = true_residuals[n_cal:]
    n_test = len(test_scores)

    # Partition calibration into blocks
    n_blocks = n_cal // block_size
    if n_blocks < 2:
        # Fall back to standard split if too few blocks
        q_hat = np.quantile(cal_scores, 1 - alpha)
        true_covered = (test_true_res <= q_hat).astype(float)
        return ConformalResult(
            coverage=true_covered.mean(),
            avg_width=2 * q_hat,
            coverages_t=true_covered,
        )

    # Trim calibration to fit exactly into blocks
    cal_trimmed = cal_scores[:n_blocks * block_size]
    blocks = cal_trimmed.reshape(n_blocks, block_size)

    # Block-permutation quantiles
    perm_quantiles = np.zeros(n_reps)
    for r in range(n_reps):
        perm_idx = rng.permutation(n_blocks)
        perm_scores = blocks[perm_idx].ravel()
        perm_quantiles[r] = np.quantile(perm_scores, 1 - alpha)

    q_hat = np.median(perm_quantiles)
    true_covered = (test_true_res <= q_hat).astype(float)

    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * q_hat,
        coverages_t=true_covered,
    )


# =========================================================================
# Method 8: HAC (Newey-West) asymptotic CIs
# =========================================================================

def hac(psi_dr_seq, tau_hat_seq, true_residuals, alpha=0.1, bandwidth=10):
    """HAC (Newey-West) asymptotic confidence intervals.

    Constructs pointwise CIs for CATE using Newey-West standard errors
    with Bartlett kernel and fixed bandwidth.

    Parameters
    ----------
    psi_dr_seq : array (T,)
        DR pseudo-outcomes (signed, not absolute value).
    tau_hat_seq : array (T,)
        CATE point estimates.
    true_residuals : array (T,)
        |tau_true - tau_hat| for true coverage.
    bandwidth : int
        Lag truncation for Bartlett kernel.
    """
    T = len(psi_dr_seq)
    residuals = psi_dr_seq - tau_hat_seq  # centering: should be mean-zero

    # Newey-West variance estimator with Bartlett kernel
    gamma_0 = np.mean(residuals**2)
    nw_var = gamma_0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)  # Bartlett kernel
        gamma_lag = np.mean(residuals[lag:] * residuals[:-lag])
        nw_var += 2 * weight * gamma_lag

    nw_se = np.sqrt(max(nw_var / T, 1e-10))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    half_width = z_crit * nw_se

    # Coverage: does true CATE fall within [tau_hat +/- half_width]?
    # This is equivalent to |tau_true - tau_hat| <= half_width
    true_covered = (true_residuals <= half_width).astype(float)

    return ConformalResult(
        coverage=true_covered.mean(),
        avg_width=2 * half_width,
        coverages_t=true_covered,
    )
