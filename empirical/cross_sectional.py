"""
Experiment 2a: Cross-sectional DR-ACI on Dynamic M-ELO.

Collapse daily panel to one observation per ticker:
  - W_i = 1 if adoption_date <= 2024-05-10, else 0
  - Y_i = mean(hidden_share) over post-cutoff window
  - X_i = pre-treatment covariate means

Run all 9 conformal methods and report coverage metrics.
"""

import numpy as np
import pandas as pd

from empirical.config import (
    ALPHA, GAMMA, TWFE_ATT,
    FIG_DIR, TAB_DIR, RESULTS_DIR,
)
from empirical.data_prep import prepare_cross_sectional, get_covariate_matrix

from draci.methods import METHODS, METHOD_LABELS
from draci.nuisance import fit_nuisances
from draci.conformal import (
    dr_score, naive_score, vs_dr_score, dr_pseudo_outcome,
    dr_aci, vs_dr_aci, split_conformal, nexcp, aci, eci,
    block_cp, hac,
    estimate_if_variance,
    ConformalResult,
)


def run_cross_sectional(cs_df: pd.DataFrame | None = None) -> dict:
    """Run cross-sectional DR-ACI experiment.

    Returns dict mapping method -> ConformalResult.
    """
    if cs_df is None:
        cs_df = prepare_cross_sectional()

    X = get_covariate_matrix(cs_df)
    W = cs_df['W'].values.astype(float)
    Y = cs_df['Y'].values.astype(float)

    n = len(X)
    n_train = int(n * 0.6)  # 60% train, 40% calibrate
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    # Split (randomized for cross-section)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    X_train, W_train, Y_train = X[train_idx], W[train_idx], Y[train_idx]
    X_cal, W_cal, Y_cal = X[cal_idx], W[cal_idx], Y[cal_idx]
    n_cal = len(X_cal)

    # Fit nuisance on training set
    e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(
        X_train, W_train, Y_train, method='xgboost',
    )
    e_hat_cal = e_hat_fn(X_cal)
    mu0_hat_cal = mu0_hat_fn(X_cal)
    mu1_hat_cal = mu1_hat_fn(X_cal)
    tau_hat_cal = tau_hat_fn(X_cal)

    # DR pseudo-outcomes
    psi_dr_cal = dr_pseudo_outcome(Y_cal, W_cal, X_cal,
                                   e_hat_cal, mu0_hat_cal, mu1_hat_cal)

    # Scores
    dr_scores = dr_score(Y_cal, W_cal, X_cal, e_hat_cal,
                         mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
    naive_scores = naive_score(Y_cal, W_cal, X_cal,
                               mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
    vs_scores = vs_dr_score(Y_cal, W_cal, X_cal, e_hat_cal,
                            mu0_hat_cal, mu1_hat_cal, tau_hat_cal)

    # For cross-section: no true CATE available
    # Use self-coverage (psi_DR in interval) as primary metric
    # Also check if TWFE ATT falls in average interval
    self_residuals = np.abs(psi_dr_cal - tau_hat_cal)

    # Variance for VS
    sigma_hat = estimate_if_variance(Y_cal, W_cal, e_hat_cal, mu0_hat_cal, mu1_hat_cal)
    self_res_std = self_residuals / sigma_hat

    n_warmup = max(20, n_cal // 10)
    half = n_cal // 2

    # Oracle: use known assignment probabilities from staggered rollout
    # Approximate: fraction treated in each cohort
    n_treated = W_cal.sum()
    n_total = len(W_cal)
    e_oracle = np.full(n_cal, n_treated / n_total)
    e_oracle = np.clip(e_oracle, 0.05, 0.95)
    oracle_scores = dr_score(Y_cal, W_cal, X_cal, e_oracle,
                             mu0_hat_cal, mu1_hat_cal, tau_hat_cal)

    results = {}

    # 1. DR-ACI
    results['dr_aci'] = dr_aci(dr_scores, self_residuals,
                               alpha=ALPHA, gamma=GAMMA, n_warmup=n_warmup)
    # 2. VS-DR-ACI
    results['vs_dr_aci'] = vs_dr_aci(vs_scores, self_res_std, self_residuals,
                                     alpha=ALPHA, gamma=GAMMA, n_warmup=n_warmup)
    # 3. Split conformal
    results['split'] = split_conformal(naive_scores[:half], naive_scores[half:],
                                       self_residuals[half:], alpha=ALPHA)
    # 4. NexCP
    results['nexcp'] = nexcp(dr_scores, self_residuals,
                             alpha=ALPHA, lam=0.05, n_warmup=n_warmup)
    # 5. ACI (non-DR)
    results['aci_no_dr'] = aci(naive_scores, self_residuals,
                               alpha=ALPHA, gamma=GAMMA, n_warmup=n_warmup)
    # 6. ECI
    results['eci'] = eci(naive_scores, self_residuals,
                         alpha=ALPHA, gamma=GAMMA, n_warmup=n_warmup)
    # 7. Block CP
    results['block_cp'] = block_cp(dr_scores, self_residuals,
                                   alpha=ALPHA, block_size=10, rng=rng)
    # 8. HAC
    results['hac'] = hac(psi_dr_cal, tau_hat_cal, self_residuals,
                         alpha=ALPHA, bandwidth=10)
    # 9. Oracle
    results['oracle'] = dr_aci(oracle_scores, self_residuals,
                               alpha=ALPHA, gamma=GAMMA, n_warmup=n_warmup)

    # Additional metric: does TWFE ATT fall in mean CATE interval?
    mean_tau_hat = np.mean(tau_hat_cal)
    att_in_interval = {}
    for m, res in results.items():
        half_w = res.avg_width / 2
        att_in_interval[m] = abs(TWFE_ATT - mean_tau_hat) <= half_w

    return results, att_in_interval, {
        'n_train': n_train, 'n_cal': n_cal,
        'mean_tau_hat': mean_tau_hat,
        'mean_psi_dr': np.mean(psi_dr_cal),
    }


def save_cs_results(results: dict, att_check: dict, meta: dict):
    """Save cross-sectional results to CSV and print summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in METHODS:
        if m in results:
            res = results[m]
            rows.append({
                'method': m,
                'label': METHOD_LABELS[m],
                'coverage': res.coverage,
                'avg_width': res.avg_width,
                'att_in_interval': att_check.get(m, None),
            })

    df = pd.DataFrame(rows)
    outpath = RESULTS_DIR / 'cs_coverage.csv'
    df.to_csv(outpath, index=False)
    print(f"\nCross-sectional results saved to {outpath}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2a: Cross-Sectional DR-ACI on M-ELO")
    print("=" * 60)

    results, att_check, meta = run_cross_sectional()
    save_cs_results(results, att_check, meta)
