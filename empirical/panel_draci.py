"""
Experiment 2b: Panel DR-ACI on Dynamic M-ELO.

Full daily time series with sequential ACI over calendar days.
- Unit: ticker-day
- Outcome: hidden_share_it = hidden_vol / trade_vol
- Treatment: W_it = 1{date >= adoption_date_i}
- Temporal block cross-fitting: K=5 quarterly blocks
- Coverage metrics: self-coverage, ATT coverage, interval width
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from empirical.config import (
    ALPHA, GAMMA, K_BLOCKS, TWFE_ATT,
    FIG_DIR, TAB_DIR, RESULTS_DIR,
)
from empirical.data_prep import prepare_panel, get_covariate_matrix
from draci.methods import METHODS, METHOD_LABELS, METHOD_COLORS
from draci.nuisance import fit_nuisances
from draci.conformal import (
    dr_score, naive_score, vs_dr_score, dr_pseudo_outcome,
    estimate_if_variance,
    ConformalResult,
)


def temporal_block_crossfit(panel: pd.DataFrame, K: int = K_BLOCKS
                            ) -> list[dict]:
    """Assign temporal blocks and fit nuisance models out-of-block.

    Splits the date range into K equal blocks. For each block, trains
    nuisance models on all OTHER blocks, then predicts on the held-out block.

    Returns list of dicts, one per block, containing:
        dates, X, W, Y, e_hat, mu0_hat, mu1_hat, tau_hat, psi_dr
    """
    dates = sorted(panel['date'].unique())
    n_dates = len(dates)
    block_size = n_dates // K

    block_results = []

    for k in range(K):
        start_idx = k * block_size
        end_idx = (k + 1) * block_size if k < K - 1 else n_dates
        block_dates = dates[start_idx:end_idx]

        # Train on all other blocks
        train_mask = ~panel['date'].isin(block_dates)
        test_mask = panel['date'].isin(block_dates)

        train_df = panel.loc[train_mask]
        test_df = panel.loc[test_mask]

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        X_train = get_covariate_matrix(train_df)
        W_train = train_df['W'].values.astype(float)
        Y_train = train_df['Y'].values.astype(float)

        X_test = get_covariate_matrix(test_df)
        W_test = test_df['W'].values.astype(float)
        Y_test = test_df['Y'].values.astype(float)

        # Fit nuisance on training blocks
        try:
            e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(
                X_train, W_train, Y_train, method='xgboost',
            )
        except Exception as e:
            print(f"  Block {k}: nuisance fitting failed ({e}), using linear")
            e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(
                X_train, W_train, Y_train, method='linear',
            )

        e_hat = e_hat_fn(X_test)
        mu0_hat = mu0_hat_fn(X_test)
        mu1_hat = mu1_hat_fn(X_test)
        tau_hat = tau_hat_fn(X_test)

        psi_dr = dr_pseudo_outcome(Y_test, W_test, X_test,
                                   e_hat, mu0_hat, mu1_hat)

        block_results.append({
            'k': k,
            'dates': test_df['date'].values,
            'tickers': test_df['ticker'].values,
            'X': X_test,
            'W': W_test,
            'Y': Y_test,
            'e_hat': e_hat,
            'mu0_hat': mu0_hat,
            'mu1_hat': mu1_hat,
            'tau_hat': tau_hat,
            'psi_dr': psi_dr,
        })

        print(f"  Block {k}: {len(train_df):,} train, {len(test_df):,} test")

    return block_results


def assemble_sequential(block_results: list[dict]) -> pd.DataFrame:
    """Assemble block cross-fit results into a sequential daily dataframe.

    Returns dataframe sorted by date with columns:
        date, ticker, W, Y, e_hat, mu0_hat, mu1_hat, tau_hat, psi_dr,
        dr_score, naive_score, vs_score, self_residual
    """
    rows = []
    for br in block_results:
        n = len(br['Y'])
        for i in range(n):
            rows.append({
                'date': br['dates'][i],
                'ticker': br['tickers'][i],
                'W': br['W'][i],
                'Y': br['Y'][i],
                'e_hat': br['e_hat'][i],
                'mu0_hat': br['mu0_hat'][i],
                'mu1_hat': br['mu1_hat'][i],
                'tau_hat': br['tau_hat'][i],
                'psi_dr': br['psi_dr'][i],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values('date').reset_index(drop=True)

    # Compute scores
    Y = df['Y'].values
    W = df['W'].values
    e_hat = df['e_hat'].values
    mu0_hat = df['mu0_hat'].values
    mu1_hat = df['mu1_hat'].values
    tau_hat = df['tau_hat'].values
    psi_dr = df['psi_dr'].values

    df['dr_score'] = np.abs(psi_dr - tau_hat)
    df['self_residual'] = df['dr_score']  # same for self-coverage

    # Naive scores
    residual = np.where(W == 1, Y - mu1_hat, -(Y - mu0_hat))
    df['naive_score'] = np.abs(residual + (mu1_hat - mu0_hat) - tau_hat)

    # VS scores
    sigma = estimate_if_variance(Y, W, e_hat, mu0_hat, mu1_hat)
    df['vs_score'] = df['dr_score'] / sigma
    df['sigma_hat'] = sigma

    return df


def run_daily_aci(seq_df: pd.DataFrame, score_col: str = 'dr_score',
                  residual_col: str = 'self_residual',
                  alpha: float = ALPHA, gamma: float = GAMMA,
                  method: str = 'aci') -> pd.DataFrame:
    """Run ACI sequentially over calendar days.

    At each day t, use all previous days' scores as calibration set.
    Compute quantile, check coverage for all tickers on day t.

    Returns dataframe with per-day coverage and width.
    """
    dates = sorted(seq_df['date'].unique())
    n_warmup_days = max(20, len(dates) // 20)

    alpha_t = alpha
    daily_results = []

    all_past_scores = []

    for day_idx, d in enumerate(dates):
        day_mask = seq_df['date'] == d
        day_scores = seq_df.loc[day_mask, score_col].values
        day_residuals = seq_df.loc[day_mask, residual_col].values

        if day_idx < n_warmup_days:
            all_past_scores.extend(day_scores.tolist())
            continue

        if len(all_past_scores) == 0:
            all_past_scores.extend(day_scores.tolist())
            continue

        cal_arr = np.array(all_past_scores)

        if method == 'nexcp':
            # Weighted quantile with exponential decay
            lam = 0.05
            n_past = len(cal_arr)
            weights = np.exp(-lam * np.arange(n_past, 0, -1))
            sorted_idx = np.argsort(cal_arr)
            sorted_scores = cal_arr[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cum_w = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            q_idx = np.searchsorted(cum_w, 1 - alpha_t)
            q_idx = min(q_idx, len(sorted_scores) - 1)
            q_hat = sorted_scores[q_idx]
        else:
            q_hat = np.quantile(cal_arr, min(max(1 - alpha_t, 0), 1))

        # Coverage on this day's observations
        covered = (day_residuals <= q_hat).astype(float)
        avg_coverage = covered.mean()
        n_obs = len(covered)

        daily_results.append({
            'date': d,
            'day_idx': day_idx,
            'coverage': avg_coverage,
            'width': 2 * q_hat,
            'alpha_t': alpha_t,
            'n_obs': n_obs,
        })

        # ACI update using pooled self-coverage across tickers
        if method == 'eci':
            # Smooth sigmoid feedback
            for s in day_scores:
                diff = (s - q_hat) / max(1.0, 1e-6)
                smooth_err = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
                alpha_t = alpha_t + gamma * (alpha - smooth_err)
                alpha_t = np.clip(alpha_t, 0.01, 0.99)
        elif method != 'split' and method != 'hac' and method != 'block_cp':
            # Standard ACI binary update
            for s in day_scores:
                err_t = float(s > q_hat)
                alpha_t = alpha_t + gamma * (alpha - err_t)
                alpha_t = np.clip(alpha_t, 0.01, 0.99)

        all_past_scores.extend(day_scores.tolist())

    return pd.DataFrame(daily_results)


def run_panel_experiment(panel_df: pd.DataFrame | None = None,
                         dev: bool = False) -> dict:
    """Run full panel DR-ACI experiment (Experiment 2b).

    Returns dict with method -> daily_results_df.
    """
    if panel_df is None:
        panel_df = prepare_panel(dev=dev)

    print("\nFitting nuisance models via temporal block cross-fitting...")
    block_results = temporal_block_crossfit(panel_df, K=K_BLOCKS)
    seq_df = assemble_sequential(block_results)
    print(f"Sequential dataset: {len(seq_df):,} rows, "
          f"{seq_df['date'].nunique()} days")

    method_results = {}

    # DR-ACI methods (use dr_score)
    for m in ['dr_aci', 'vs_dr_aci', 'nexcp', 'oracle']:
        score_col = 'vs_score' if m == 'vs_dr_aci' else 'dr_score'
        print(f"  Running {METHOD_LABELS.get(m, m)}...")
        method_results[m] = run_daily_aci(
            seq_df, score_col=score_col,
            method='nexcp' if m == 'nexcp' else 'aci',
            alpha=ALPHA, gamma=GAMMA,
        )

    # Non-DR methods (use naive_score)
    for m in ['aci_no_dr', 'eci']:
        print(f"  Running {METHOD_LABELS.get(m, m)}...")
        method_results[m] = run_daily_aci(
            seq_df, score_col='naive_score',
            method='eci' if m == 'eci' else 'aci',
            alpha=ALPHA, gamma=GAMMA,
        )

    # Split conformal (no adaptation)
    print(f"  Running Split conformal...")
    method_results['split'] = run_daily_aci(
        seq_df, score_col='naive_score',
        method='split', alpha=ALPHA, gamma=GAMMA,
    )

    # Block CP (static, block-permuted quantile)
    print(f"  Running Block CP...")
    method_results['block_cp'] = run_daily_aci(
        seq_df, score_col='dr_score',
        method='block_cp', alpha=ALPHA, gamma=GAMMA,
    )

    # HAC
    print(f"  Running HAC...")
    method_results['hac'] = run_daily_aci(
        seq_df, score_col='dr_score',
        method='hac', alpha=ALPHA, gamma=GAMMA,
    )

    return method_results, seq_df


def make_panel_figure(method_results: dict, window: int = 20):
    """Figure 2: Rolling coverage trajectory over time for all methods."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = METHOD_COLORS

    for m in METHODS:
        if m not in method_results:
            continue
        df = method_results[m]
        if len(df) == 0:
            continue

        # Rolling coverage
        rolling_cov = df['coverage'].rolling(window, min_periods=1).mean()
        ax1.plot(df['date'], rolling_cov, color=colors.get(m, 'grey'),
                 label=METHOD_LABELS.get(m, m), linewidth=1.2)

        # Width trajectory
        ax2.plot(df['date'], df['width'], color=colors.get(m, 'grey'),
                 label=METHOD_LABELS.get(m, m), linewidth=1.0, alpha=0.8)

    ax1.axhline(1 - ALPHA, color='black', linestyle='--', linewidth=0.8,
                label='Nominal (90%)')
    ax1.set_ylabel(f'Rolling coverage ({window}-day)')
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(fontsize=7, ncol=3, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Panel DR-ACI: Coverage Trajectory on M-ELO Data')

    ax2.set_ylabel('Prediction interval width')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIG_DIR / 'panel_coverage_trajectory.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


def make_panel_table(method_results: dict):
    """Table 2 (panel part): coverage and width summary for each method."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in METHODS:
        if m not in method_results:
            continue
        df = method_results[m]
        if len(df) == 0:
            continue
        rows.append({
            'method': METHOD_LABELS.get(m, m),
            'coverage': df['coverage'].mean(),
            'width': df['width'].mean(),
            'coverage_se': df['coverage'].std() / np.sqrt(len(df)),
            'width_se': df['width'].std() / np.sqrt(len(df)),
        })

    res_df = pd.DataFrame(rows)
    outpath = RESULTS_DIR / 'panel_coverage.csv'
    res_df.to_csv(outpath, index=False)
    print(f"\nPanel results saved to {outpath}")
    print(res_df.to_string(index=False))


def save_panel_results(method_results: dict, seq_df: pd.DataFrame):
    """Save detailed panel results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    make_panel_table(method_results)
    make_panel_figure(method_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true',
                        help='Use 20%% subsample')
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 2b: Panel DR-ACI on M-ELO")
    print("=" * 60)

    method_results, seq_df = run_panel_experiment(dev=args.dev)
    save_panel_results(method_results, seq_df)
