"""
Experiment 4: Sensitivity analysis (Appendix).

Varies hyperparameters on the panel DR-ACI:
  - gamma: {0.001, 0.005, 0.01, 0.02, 0.05}
  - K (temporal blocks): {3, 5, 7, 10}
  - gap_frac: {0.0, 0.05, 0.10}

Reports coverage vs width tradeoff for each combination.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from empirical.config import (
    ALPHA, SENSITIVITY_GAMMAS, SENSITIVITY_K, SENSITIVITY_GAP_FRAC,
    FIG_DIR, TAB_DIR, RESULTS_DIR,
)
from empirical.data_prep import prepare_panel
from empirical.panel_draci import temporal_block_crossfit, assemble_sequential, run_daily_aci


def run_sensitivity(panel_df: pd.DataFrame | None = None,
                    dev: bool = True) -> pd.DataFrame:
    """Run sensitivity analysis over (gamma, K, gap_frac) grid.

    Returns dataframe with columns:
        gamma, K, gap_frac, coverage, width
    """
    if panel_df is None:
        panel_df = prepare_panel(dev=dev)

    results = []

    # Main sensitivity: vary gamma with K=5 (baseline)
    print("\n--- Sensitivity: varying gamma ---")
    block_results = temporal_block_crossfit(panel_df, K=5)
    seq_df = assemble_sequential(block_results)

    for gamma in SENSITIVITY_GAMMAS:
        print(f"  gamma={gamma}")
        daily = run_daily_aci(seq_df, score_col='dr_score',
                              method='aci', alpha=ALPHA, gamma=gamma)
        results.append({
            'param': 'gamma', 'value': gamma,
            'gamma': gamma, 'K': 5, 'gap_frac': 0.0,
            'coverage': daily['coverage'].mean(),
            'width': daily['width'].mean(),
        })

    # Vary K
    print("\n--- Sensitivity: varying K ---")
    for K in SENSITIVITY_K:
        print(f"  K={K}")
        try:
            block_res_k = temporal_block_crossfit(panel_df, K=K)
            seq_df_k = assemble_sequential(block_res_k)
            daily = run_daily_aci(seq_df_k, score_col='dr_score',
                                  method='aci', alpha=ALPHA, gamma=0.005)
            results.append({
                'param': 'K', 'value': K,
                'gamma': 0.005, 'K': K, 'gap_frac': 0.0,
                'coverage': daily['coverage'].mean(),
                'width': daily['width'].mean(),
            })
        except Exception as e:
            print(f"    Failed: {e}")

    results_df = pd.DataFrame(results)
    return results_df


def make_sensitivity_table(results_df: pd.DataFrame):
    """Appendix table: sensitivity analysis results."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity analysis: coverage and PIAW under "
                 r"varying hyperparameters.}")
    lines.append(r"\label{tab:sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Value & Coverage & PIAW \\")
    lines.append(r"\midrule")

    for _, row in results_df.iterrows():
        param = row['param']
        if param == 'gamma':
            val_str = f"$\\gamma = {row['value']}$"
        elif param == 'K':
            val_str = f"$K = {int(row['value'])}$"
        else:
            val_str = f"gap = {row['value']}"

        lines.append(
            f"{param} & {val_str} & {row['coverage']:.3f} "
            f"& {row['width']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    outpath = TAB_DIR / "sensitivity.tex"
    outpath.write_text(tex)
    print(f"Saved: {outpath}")


def make_sensitivity_figure(results_df: pd.DataFrame):
    """Appendix figure: sensitivity heatmap."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Coverage vs gamma
    gamma_df = results_df[results_df['param'] == 'gamma']
    if len(gamma_df) > 0:
        ax1.plot(gamma_df['value'], gamma_df['coverage'], 'bo-', label='Coverage')
        ax1_tw = ax1.twinx()
        ax1_tw.plot(gamma_df['value'], gamma_df['width'], 'rs--', label='PIAW')
        ax1.set_xlabel(r'$\gamma$')
        ax1.set_ylabel('Coverage', color='blue')
        ax1_tw.set_ylabel('PIAW', color='red')
        ax1.axhline(0.9, color='grey', linestyle=':', alpha=0.5)
        ax1.set_title(r'Sensitivity to $\gamma$')
        ax1.grid(True, alpha=0.3)

    # Coverage vs K
    k_df = results_df[results_df['param'] == 'K']
    if len(k_df) > 0:
        ax2.plot(k_df['value'], k_df['coverage'], 'bo-', label='Coverage')
        ax2_tw = ax2.twinx()
        ax2_tw.plot(k_df['value'], k_df['width'], 'rs--', label='PIAW')
        ax2.set_xlabel('K (temporal blocks)')
        ax2.set_ylabel('Coverage', color='blue')
        ax2_tw.set_ylabel('PIAW', color='red')
        ax2.axhline(0.9, color='grey', linestyle=':', alpha=0.5)
        ax2.set_title('Sensitivity to K')
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = FIG_DIR / 'sensitivity_heatmap.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


def run_experiment_4(dev: bool = True) -> pd.DataFrame:
    """Run complete sensitivity analysis experiment."""
    results_df = run_sensitivity(dev=dev)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS_DIR / 'sensitivity.csv'
    results_df.to_csv(outpath, index=False)
    print(f"Saved raw results: {outpath}")

    make_sensitivity_table(results_df)
    make_sensitivity_figure(results_df)

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 4: Sensitivity Analysis (Appendix)")
    print("=" * 60)
    run_experiment_4(dev=not args.full)
