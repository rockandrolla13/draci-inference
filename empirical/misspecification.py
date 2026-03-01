"""
Experiment 5: Misspecification robustness (Appendix).

Demonstrates the doubly robust property empirically:
  (a) Both correct: XGBoost propensity + GBT outcome → full coverage
  (b) Propensity misspecified (constant e=0.5), outcome correct → preserved
  (c) Outcome misspecified (linear only), propensity correct → preserved
  (d) Both misspecified → coverage degrades (product-bias tax)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from empirical.config import (
    ALPHA, GAMMA, K_BLOCKS,
    FIG_DIR, TAB_DIR, RESULTS_DIR,
)
from empirical.data_prep import prepare_panel, get_covariate_matrix
from empirical.panel_draci import temporal_block_crossfit, assemble_sequential, run_daily_aci
from draci.methods import METHODS, METHOD_LABELS
from draci.conformal import dr_pseudo_outcome
from draci.nuisance import fit_nuisances, fit_nuisance_linear


def run_misspecification(panel_df: pd.DataFrame | None = None,
                         dev: bool = True) -> pd.DataFrame:
    """Run misspecification experiment with 4 configurations.

    Returns dataframe with columns:
        config, config_label, coverage, width
    """
    if panel_df is None:
        panel_df = prepare_panel(dev=dev)

    configs = {
        'both_correct': {
            'label': '(a) Both correct',
            'propensity': 'xgboost',
            'outcome': 'xgboost',
        },
        'prop_misspec': {
            'label': '(b) Propensity misspecified',
            'propensity': 'constant',
            'outcome': 'xgboost',
        },
        'outcome_misspec': {
            'label': '(c) Outcome misspecified',
            'propensity': 'xgboost',
            'outcome': 'linear',
        },
        'both_misspec': {
            'label': '(d) Both misspecified',
            'propensity': 'constant',
            'outcome': 'linear',
        },
    }

    results = []

    for config_key, config in configs.items():
        print(f"\n--- Config: {config['label']} ---")

        # Custom nuisance fitting per block
        dates = sorted(panel_df['date'].unique())
        n_dates = len(dates)
        block_size = n_dates // K_BLOCKS

        all_rows = []

        for k in range(K_BLOCKS):
            start_idx = k * block_size
            end_idx = (k + 1) * block_size if k < K_BLOCKS - 1 else n_dates
            block_dates = dates[start_idx:end_idx]

            train_mask = ~panel_df['date'].isin(block_dates)
            test_mask = panel_df['date'].isin(block_dates)

            train_df = panel_df.loc[train_mask]
            test_df = panel_df.loc[test_mask]

            if len(train_df) < 100 or len(test_df) < 10:
                continue

            X_train = get_covariate_matrix(train_df)
            W_train = train_df['W'].values.astype(float)
            Y_train = train_df['Y'].values.astype(float)
            X_test = get_covariate_matrix(test_df)
            W_test = test_df['W'].values.astype(float)
            Y_test = test_df['Y'].values.astype(float)

            # Propensity
            if config['propensity'] == 'constant':
                e_hat = np.full(len(X_test), 0.5)
            else:
                e_fn, _, _, _ = fit_nuisances(
                    X_train, W_train, Y_train, method='xgboost')
                e_hat = e_fn(X_test)

            # Outcome
            if config['outcome'] == 'linear':
                _, mu0_fn, mu1_fn, tau_fn = fit_nuisance_linear(
                    X_train, W_train, Y_train)
            else:
                _, mu0_fn, mu1_fn, tau_fn = fit_nuisances(
                    X_train, W_train, Y_train, method='xgboost')

            mu0_hat = mu0_fn(X_test)
            mu1_hat = mu1_fn(X_test)
            tau_hat = tau_fn(X_test)

            # DR pseudo-outcome
            psi_dr = dr_pseudo_outcome(Y_test, W_test, X_test, e_hat, mu0_hat, mu1_hat)
            dr_scores = np.abs(psi_dr - tau_hat)

            for i in range(len(X_test)):
                all_rows.append({
                    'date': test_df.iloc[i]['date'],
                    'ticker': test_df.iloc[i]['ticker'],
                    'dr_score': dr_scores[i],
                    'self_residual': dr_scores[i],
                })

        seq_df = pd.DataFrame(all_rows).sort_values('date').reset_index(drop=True)

        # Run DR-ACI
        daily = run_daily_aci(seq_df, score_col='dr_score',
                              method='aci', alpha=ALPHA, gamma=GAMMA)

        coverage = daily['coverage'].mean()
        width = daily['width'].mean()

        results.append({
            'config': config_key,
            'label': config['label'],
            'coverage': coverage,
            'width': width,
        })

        print(f"  Coverage: {coverage:.3f}, Width: {width:.3f}")

    return pd.DataFrame(results)


def make_misspecification_table(results_df: pd.DataFrame):
    """Appendix table: misspecification robustness results."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Misspecification robustness of DR-ACI. "
                 r"Coverage and PIAW under different nuisance "
                 r"specification regimes.}")
    lines.append(r"\label{tab:misspec}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Propensity & Outcome & Coverage & PIAW \\")
    lines.append(r"\midrule")

    prop_labels = {
        'both_correct': r'\checkmark',
        'prop_misspec': r'\texttimes',
        'outcome_misspec': r'\checkmark',
        'both_misspec': r'\texttimes',
    }
    out_labels = {
        'both_correct': r'\checkmark',
        'prop_misspec': r'\checkmark',
        'outcome_misspec': r'\texttimes',
        'both_misspec': r'\texttimes',
    }

    for _, row in results_df.iterrows():
        conf = row['config']
        lines.append(
            f"{row['label']} & {prop_labels.get(conf, '--')} "
            f"& {out_labels.get(conf, '--')} "
            f"& {row['coverage']:.3f} & {row['width']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    outpath = TAB_DIR / "misspecification.tex"
    outpath.write_text(tex)
    print(f"Saved: {outpath}")


def make_misspecification_figure(results_df: pd.DataFrame):
    """Appendix figure: misspecification bar chart."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = results_df['label'].values
    x = np.arange(len(labels))
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d6604d']

    # Coverage bars
    ax1.bar(x, results_df['coverage'], color=colors, edgecolor='white')
    ax1.axhline(0.9, color='black', linestyle='--', linewidth=0.8,
                label='Nominal (90%)')
    ax1.axhline(0.85, color='red', linestyle=':', linewidth=0.8,
                alpha=0.5, label='Threshold (85%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage under misspecification')
    ax1.legend(fontsize=7)
    ax1.set_ylim(0.6, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Width bars
    ax2.bar(x, results_df['width'], color=colors, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('PIAW')
    ax2.set_title('Interval width under misspecification')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    outpath = FIG_DIR / 'misspecification_bars.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


def run_experiment_5(dev: bool = True) -> pd.DataFrame:
    """Run complete misspecification experiment."""
    results_df = run_misspecification(dev=dev)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS_DIR / 'misspecification.csv'
    results_df.to_csv(outpath, index=False)
    print(f"Saved raw results: {outpath}")

    make_misspecification_table(results_df)
    make_misspecification_figure(results_df)

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 5: Misspecification Robustness (Appendix)")
    print("=" * 60)
    run_experiment_5(dev=not args.full)
