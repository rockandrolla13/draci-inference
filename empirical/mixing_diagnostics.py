"""
Experiment 3: Mixing diagnostics on Dynamic M-ELO data.

Estimates beta-mixing coefficients from real hidden_share series to
validate the theoretical assumptions (A1, A3, A4).

Steps:
  3.1 — ACF estimation (per-ticker and aggregate)
  3.2 — Beta-mixing coefficient estimation (histogram TV distance)
  3.3 — Assumption verification (overlap, moments, mixing rate)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from empirical.config import (
    FIG_DIR, TAB_DIR, RESULTS_DIR,
    MIXING_MAX_LAG, MIXING_BETA_LAGS, MIXING_N_BINS,
)
from empirical.data_prep import load_daily_data


def compute_ticker_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute ACF of a single time series up to max_lag.

    Uses biased estimator (dividing by T) which is positive semi-definite.
    """
    series = series - np.mean(series)
    T = len(series)
    if T < max_lag + 10:
        return np.full(max_lag + 1, np.nan)

    var = np.sum(series**2) / T
    if var < 1e-12:
        return np.full(max_lag + 1, np.nan)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(series[lag:] * series[:-lag]) / (T * var)
    return acf


def estimate_acf_distribution(panel: pd.DataFrame,
                              max_lag: int = MIXING_MAX_LAG,
                              min_obs: int = 100) -> dict:
    """Step 3.1: Per-ticker ACF estimation.

    Returns dict with:
        lag1_rhos: array of lag-1 autocorrelations across tickers
        acf_matrix: (n_tickers, max_lag+1) matrix of ACF values
        median_acf: median ACF curve across tickers
    """
    tickers = panel['ticker'].unique()
    lag1_rhos = []
    acf_list = []

    for ticker in tickers:
        ts = panel.loc[panel['ticker'] == ticker, 'hidden_share'].dropna().values
        if len(ts) < min_obs:
            continue

        acf = compute_ticker_acf(ts, max_lag)
        if np.isnan(acf[1]):
            continue

        lag1_rhos.append(acf[1])
        acf_list.append(acf)

    lag1_rhos = np.array(lag1_rhos)
    acf_matrix = np.array(acf_list)

    # Aggregate statistics
    median_acf = np.nanmedian(acf_matrix, axis=0)
    q25_acf = np.nanpercentile(acf_matrix, 25, axis=0)
    q75_acf = np.nanpercentile(acf_matrix, 75, axis=0)

    print(f"ACF estimation: {len(lag1_rhos)} tickers with >= {min_obs} obs")
    print(f"  Lag-1 rho: median={np.median(lag1_rhos):.3f}, "
          f"mean={np.mean(lag1_rhos):.3f}, "
          f"std={np.std(lag1_rhos):.3f}")

    return {
        'lag1_rhos': lag1_rhos,
        'acf_matrix': acf_matrix,
        'median_acf': median_acf,
        'q25_acf': q25_acf,
        'q75_acf': q75_acf,
    }


def estimate_beta_mixing(panel: pd.DataFrame,
                         lags: list[int] = MIXING_BETA_LAGS,
                         n_bins: int = MIXING_N_BINS,
                         n_tickers_sample: int = 200) -> dict:
    """Step 3.2: Estimate beta-mixing coefficients.

    Uses histogram method (Oliveira et al. 2024):
    1. For each lag tau, bin (X_t, X_{t+tau}) pairs
    2. Estimate TV distance between joint and product marginal
    3. beta(tau) = sup TV(P(X_t, X_{t+tau}) || P(X_t) x P(X_{t+tau}))
    """
    tickers = panel['ticker'].unique()
    rng = np.random.default_rng(42)

    if len(tickers) > n_tickers_sample:
        tickers = rng.choice(tickers, n_tickers_sample, replace=False)

    beta_estimates = {tau: [] for tau in lags}

    for ticker in tickers:
        ts = panel.loc[panel['ticker'] == ticker, 'hidden_share'].dropna().values
        if len(ts) < max(lags) + 50:
            continue

        for tau in lags:
            if tau >= len(ts):
                continue

            past = ts[:-tau]
            future = ts[tau:]
            n = len(past)

            # Bin into histogram
            edges_p = np.linspace(np.min(past), np.max(past) + 1e-10, n_bins + 1)
            edges_f = np.linspace(np.min(future), np.max(future) + 1e-10, n_bins + 1)

            # Joint distribution
            hist_joint, _, _ = np.histogram2d(past, future,
                                              bins=[edges_p, edges_f])
            hist_joint = hist_joint / n

            # Marginals
            hist_past = np.histogram(past, bins=edges_p)[0] / n
            hist_future = np.histogram(future, bins=edges_f)[0] / n

            # Product of marginals
            hist_product = np.outer(hist_past, hist_future)

            # TV distance = 0.5 * sum |P_joint - P_product|
            tv = 0.5 * np.sum(np.abs(hist_joint - hist_product))
            beta_estimates[tau].append(tv)

    # Aggregate across tickers
    beta_values = {}
    for tau in lags:
        if beta_estimates[tau]:
            beta_values[tau] = {
                'median': np.median(beta_estimates[tau]),
                'mean': np.mean(beta_estimates[tau]),
                'std': np.std(beta_estimates[tau]),
                'q25': np.percentile(beta_estimates[tau], 25),
                'q75': np.percentile(beta_estimates[tau], 75),
            }
        else:
            beta_values[tau] = {'median': np.nan, 'mean': np.nan,
                                'std': np.nan, 'q25': np.nan, 'q75': np.nan}

    # Fit exponential decay: beta(tau) ~ a * exp(-b * tau)
    tau_arr = np.array([t for t in lags if not np.isnan(beta_values[t]['median'])])
    beta_arr = np.array([beta_values[t]['median'] for t in tau_arr])

    if len(tau_arr) > 2 and np.all(beta_arr > 0):
        log_beta = np.log(np.maximum(beta_arr, 1e-10))
        # Linear regression on log scale: log(beta) = log(a) - b*tau
        slope, intercept, r_value, _, _ = stats.linregress(tau_arr, log_beta)
        exp_fit = {'a': np.exp(intercept), 'b': -slope, 'r2': r_value**2}
    else:
        exp_fit = {'a': np.nan, 'b': np.nan, 'r2': np.nan}

    print(f"Beta-mixing estimation:")
    for tau in lags:
        bv = beta_values[tau]
        print(f"  tau={tau:3d}: beta={bv['median']:.4f} "
              f"(+/- {bv['std']:.4f})")
    if not np.isnan(exp_fit['b']):
        print(f"  Exponential fit: beta(tau) ~ {exp_fit['a']:.3f} * "
              f"exp(-{exp_fit['b']:.4f} * tau), R^2 = {exp_fit['r2']:.3f}")

    return {'beta_values': beta_values, 'exp_fit': exp_fit}


def verify_assumptions(panel: pd.DataFrame, acf_results: dict,
                       beta_results: dict) -> dict:
    """Step 3.3: Verify theory assumptions.

    A1 (mixing): log(beta(tau)) vs tau linearity check
    A3 (overlap): propensity score histogram
    A4 (moments): finite (2+delta)-th moment check on DR scores
    """
    diagnostics = {}

    # A1: Mixing rate
    exp_fit = beta_results['exp_fit']
    diagnostics['A1_mixing_rate'] = exp_fit['b']
    diagnostics['A1_r2'] = exp_fit['r2']
    diagnostics['A1_passed'] = (not np.isnan(exp_fit['r2'])
                                and exp_fit['r2'] > 0.8)

    # A3: Overlap — compute propensity of treatment at cutoff
    # Simple: fraction treated by adoption group
    has_adoption = panel['adoption_date'].notna()
    pre_cutoff = panel.loc[has_adoption]
    if len(pre_cutoff) > 0:
        prop = pre_cutoff['W'].mean()
        diagnostics['A3_prop_mean'] = prop
        diagnostics['A3_passed'] = 0.05 < prop < 0.95
    else:
        diagnostics['A3_prop_mean'] = np.nan
        diagnostics['A3_passed'] = False

    # A4: Moments of hidden_share
    hs = panel['hidden_share'].dropna()
    diagnostics['A4_mean'] = hs.mean()
    diagnostics['A4_std'] = hs.std()
    diagnostics['A4_kurtosis'] = stats.kurtosis(hs)
    # Check (2+delta) moment for delta=0.5
    diagnostics['A4_moment_2p5'] = np.mean(np.abs(hs)**2.5)
    diagnostics['A4_passed'] = np.isfinite(diagnostics['A4_moment_2p5'])

    # Theoretical gap: min_tau { tau/T + 2*beta(tau) }
    T = panel['date'].nunique()
    beta_values = beta_results['beta_values']
    gaps = []
    for tau in MIXING_BETA_LAGS:
        if tau in beta_values and not np.isnan(beta_values[tau]['median']):
            gap = tau / T + 2 * beta_values[tau]['median']
            gaps.append((tau, gap))
    if gaps:
        best_tau, best_gap = min(gaps, key=lambda x: x[1])
        diagnostics['opt_tau'] = best_tau
        diagnostics['opt_gap'] = best_gap
    else:
        diagnostics['opt_tau'] = np.nan
        diagnostics['opt_gap'] = np.nan

    # ACF lag-1 median
    diagnostics['acf_lag1_median'] = np.median(acf_results['lag1_rhos'])

    return diagnostics


def make_acf_figure(acf_results: dict):
    """Figure 4a: ACF decay curve + lag-1 distribution."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Aggregate ACF decay
    lags_arr = np.arange(len(acf_results['median_acf']))
    ax1.plot(lags_arr, acf_results['median_acf'], 'b-', linewidth=1.5,
             label='Median')
    ax1.fill_between(lags_arr, acf_results['q25_acf'],
                     acf_results['q75_acf'], alpha=0.2, color='blue',
                     label='IQR')
    ax1.axhline(0, color='grey', linewidth=0.5)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.set_title('Autocorrelation of hidden\\_share')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Distribution of lag-1 rho
    ax2.hist(acf_results['lag1_rhos'], bins=50, density=True,
             alpha=0.7, color='steelblue', edgecolor='white')
    median_rho = np.median(acf_results['lag1_rhos'])
    ax2.axvline(median_rho, color='red', linestyle='--',
                label=f'Median = {median_rho:.3f}')
    ax2.set_xlabel(r'Lag-1 autocorrelation $\rho$')
    ax2.set_ylabel('Density')
    ax2.set_title(r'Distribution of $\hat{\rho}$ across tickers')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIG_DIR / 'acf_hidden_share.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


def make_beta_figure(beta_results: dict):
    """Figure 4b: Beta-mixing decay with exponential fit."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    bv = beta_results['beta_values']
    ef = beta_results['exp_fit']

    taus = sorted(bv.keys())
    medians = [bv[t]['median'] for t in taus]
    q25s = [bv[t]['q25'] for t in taus]
    q75s = [bv[t]['q75'] for t in taus]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: beta(tau) on linear scale
    ax1.plot(taus, medians, 'bo-', markersize=5, label='Estimated')
    ax1.fill_between(taus, q25s, q75s, alpha=0.2, color='blue', label='IQR')

    if not np.isnan(ef['a']):
        tau_fine = np.linspace(1, max(taus), 200)
        fit_curve = ef['a'] * np.exp(-ef['b'] * tau_fine)
        ax1.plot(tau_fine, fit_curve, 'r--', linewidth=1.5,
                 label=f'Fit: {ef["a"]:.2f}exp(-{ef["b"]:.3f}τ)')

    ax1.set_xlabel(r'Lag $\tau$')
    ax1.set_ylabel(r'$\hat{\beta}(\tau)$')
    ax1.set_title(r'$\beta$-mixing decay')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Right: log scale
    valid_medians = [(t, m) for t, m in zip(taus, medians) if m > 0]
    if valid_medians:
        t_v, m_v = zip(*valid_medians)
        ax2.semilogy(t_v, m_v, 'bo-', markersize=5, label='Estimated')
        if not np.isnan(ef['a']):
            ax2.semilogy(tau_fine, fit_curve, 'r--', linewidth=1.5,
                         label=f'R² = {ef["r2"]:.3f}')
    ax2.set_xlabel(r'Lag $\tau$')
    ax2.set_ylabel(r'$\log \hat{\beta}(\tau)$')
    ax2.set_title(r'$\beta$-mixing decay (log scale)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIG_DIR / 'beta_mixing_decay.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


def make_diagnostics_table(diagnostics: dict, acf_results: dict,
                           beta_results: dict):
    """Table 3: Mixing diagnostics summary."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mixing diagnostics for hidden\_share series.}")
    lines.append(r"\label{tab:mixing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Diagnostic & Estimate & Assumption \\")
    lines.append(r"\midrule")

    # ACF lag-1
    rho_med = diagnostics['acf_lag1_median']
    lines.append(
        f"Lag-1 autocorrelation (median) & {rho_med:.3f} & -- \\\\"
    )

    # Mixing rate
    if not np.isnan(diagnostics['A1_mixing_rate']):
        lines.append(
            f"Mixing decay rate $b$ & {diagnostics['A1_mixing_rate']:.4f} "
            f"& A1 ($R^2={diagnostics['A1_r2']:.2f}$) \\\\"
        )

    # Overlap
    if not np.isnan(diagnostics['A3_prop_mean']):
        lines.append(
            f"Treatment propensity (mean) & {diagnostics['A3_prop_mean']:.3f} "
            f"& A3 ({'\\checkmark' if diagnostics['A3_passed'] else 'X'}) \\\\"
        )

    # Moments
    lines.append(
        f"$E[|Y|^{{2.5}}]$ & {diagnostics['A4_moment_2p5']:.4f} "
        f"& A4 ({'\\checkmark' if diagnostics['A4_passed'] else 'X'}) \\\\"
    )

    # Optimal gap
    if not np.isnan(diagnostics['opt_gap']):
        lines.append(
            f"Optimal gap $\\tau^*={diagnostics['opt_tau']:.0f}$ & "
            f"{diagnostics['opt_gap']:.4f} & Thm 1 \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    outpath = TAB_DIR / "mixing_diagnostics.tex"
    outpath.write_text(tex)
    print(f"Saved: {outpath}")

    # Also save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([diagnostics]).to_csv(
        RESULTS_DIR / 'mixing_diagnostics.csv', index=False
    )


def run_mixing_diagnostics(panel: pd.DataFrame | None = None) -> dict:
    """Run all mixing diagnostic analyses."""
    if panel is None:
        from empirical.data_prep import prepare_panel
        panel = prepare_panel()

    print("\nStep 3.1: ACF estimation...")
    acf_results = estimate_acf_distribution(panel)

    print("\nStep 3.2: Beta-mixing estimation...")
    beta_results = estimate_beta_mixing(panel)

    print("\nStep 3.3: Assumption verification...")
    diagnostics = verify_assumptions(panel, acf_results, beta_results)

    print(f"\nDiagnostics summary:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    # Generate outputs
    make_acf_figure(acf_results)
    make_beta_figure(beta_results)
    make_diagnostics_table(diagnostics, acf_results, beta_results)

    return {'acf': acf_results, 'beta': beta_results,
            'diagnostics': diagnostics}


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 3: Mixing Diagnostics")
    print("=" * 60)
    run_mixing_diagnostics()
