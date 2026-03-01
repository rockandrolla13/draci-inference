"""
Monte Carlo simulation: DR-ACI coverage under beta-mixing (Section 5).

DGP: 5D AR(1) covariates with GARCH(1,1) errors.
Nine conformal methods compared across varying mixing strength rho.

Usage:
    python sim_coverage.py                 # full production run (500 MC)
    python sim_coverage.py --quick         # fast dev run (10 MC, 2 rho)
    python sim_coverage.py --linear        # use linear nuisance (faster)
    python sim_coverage.py --n-jobs 8      # parallel MC trials

Output:
    paper/figures/coverage_vs_mixing.pdf  -- Figure 1
    paper/tables/coverage_table.tex       -- Table 1
    results/sim_raw.csv                   -- raw MC results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import csv

from simulation.config import (
    get_config, parse_sim_args, METHODS, METHOD_LABELS,
    TABLE_RHOS, FIG_DIR, TAB_DIR, RESULTS_DIR,
    BLOCK_SIZE, NEXCP_LAMBDA, N_WARMUP_FRAC,
)
from draci.dgp import generate_data
from draci.nuisance import fit_nuisances
from draci.conformal import (
    dr_score, naive_score, vs_dr_score, dr_pseudo_outcome,
    estimate_if_variance,
    dr_aci, vs_dr_aci, split_conformal, nexcp, aci, eci,
    block_cp, hac,
)
from draci.methods import METHOD_COLORS, METHOD_MARKERS


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------
def run_one_trial(T: int, rho: float, seed: int,
                  cfg: dict, nuisance_method: str = 'xgboost') -> dict:
    """Run one MC trial with all 9 methods.

    Parameters
    ----------
    seed : int
        Per-trial seed. Creates its own RNG internally so that
        the function is safe for joblib parallel dispatch.

    Returns dict: method_name -> (coverage, avg_width).
    """
    rng = np.random.default_rng(seed)
    data = generate_data(T, rho, rng)
    X, W, Y = data['X'], data['W'], data['Y']
    e_true = data['e_true']
    mu0_true, mu1_true = data['mu0_true'], data['mu1_true']
    tau_true = data['tau_true']

    alpha = cfg['alpha']
    gamma = cfg['gamma']
    train_frac = cfg['train_frac']
    gap_frac = cfg['gap_frac']

    n_train = int(T * train_frac)
    n_gap = int(T * gap_frac)
    cal_start = n_train + n_gap

    X_train, W_train, Y_train = X[:n_train], W[:n_train], Y[:n_train]
    X_cal, W_cal, Y_cal = X[cal_start:], W[cal_start:], Y[cal_start:]
    tau_cal = tau_true[cal_start:]
    e_true_cal = e_true[cal_start:]
    mu0_true_cal = mu0_true[cal_start:]
    mu1_true_cal = mu1_true[cal_start:]

    n_cal = len(X_cal)
    n_warmup = max(int(n_cal * N_WARMUP_FRAC), 20)

    # --- Fit nuisances on training block ---
    e_hat_fn, mu0_hat_fn, mu1_hat_fn, tau_hat_fn = fit_nuisances(
        X_train, W_train, Y_train, method=nuisance_method,
    )
    e_hat_cal = e_hat_fn(X_cal)
    mu0_hat_cal = mu0_hat_fn(X_cal)
    mu1_hat_cal = mu1_hat_fn(X_cal)
    tau_hat_cal = tau_hat_fn(X_cal)

    # --- True residuals: |tau_true - tau_hat| ---
    true_residuals = np.abs(tau_cal - tau_hat_cal)

    # --- Compute scores ---
    dr_scores_est = dr_score(Y_cal, W_cal, X_cal, e_hat_cal,
                             mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
    naive_scores_est = naive_score(Y_cal, W_cal, X_cal,
                                   mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
    vs_scores_est = vs_dr_score(Y_cal, W_cal, X_cal, e_hat_cal,
                                mu0_hat_cal, mu1_hat_cal, tau_hat_cal)
    psi_dr_est = dr_pseudo_outcome(Y_cal, W_cal, X_cal, e_hat_cal,
                                   mu0_hat_cal, mu1_hat_cal)

    # Oracle: true nuisances, estimated CATE
    oracle_scores = dr_score(Y_cal, W_cal, X_cal, e_true_cal,
                             mu0_true_cal, mu1_true_cal, tau_hat_cal)
    oracle_psi = dr_pseudo_outcome(Y_cal, W_cal, X_cal, e_true_cal,
                                   mu0_true_cal, mu1_true_cal)

    # --- Variance for VS-DR-ACI ---
    sigma_hat = estimate_if_variance(Y_cal, W_cal, e_hat_cal,
                                     mu0_hat_cal, mu1_hat_cal)
    true_res_standardized = true_residuals / sigma_hat

    results = {}

    # 1. DR-ACI
    res = dr_aci(dr_scores_est, true_residuals,
                 alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['dr_aci'] = (res.coverage, res.avg_width)

    # 2. VS-DR-ACI
    res = vs_dr_aci(vs_scores_est, true_res_standardized, true_residuals,
                    alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['vs_dr_aci'] = (res.coverage, res.avg_width)

    # 3. Split conformal
    half = n_cal // 2
    res = split_conformal(naive_scores_est[:half], naive_scores_est[half:],
                          true_residuals[half:], alpha=alpha)
    results['split'] = (res.coverage, res.avg_width)

    # 4. NexCP
    res = nexcp(dr_scores_est, true_residuals,
                alpha=alpha, lam=NEXCP_LAMBDA, n_warmup=n_warmup)
    results['nexcp'] = (res.coverage, res.avg_width)

    # 5. ACI (non-DR)
    res = aci(naive_scores_est, true_residuals,
              alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['aci_no_dr'] = (res.coverage, res.avg_width)

    # 6. ECI
    res = eci(naive_scores_est, true_residuals,
              alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['eci'] = (res.coverage, res.avg_width)

    # 7. Block CP
    res = block_cp(dr_scores_est, true_residuals,
                   alpha=alpha, block_size=BLOCK_SIZE, rng=rng)
    results['block_cp'] = (res.coverage, res.avg_width)

    # 8. HAC (Newey-West)
    res = hac(psi_dr_est, tau_hat_cal, true_residuals,
              alpha=alpha, bandwidth=10)
    results['hac'] = (res.coverage, res.avg_width)

    # 9. Oracle DR-ACI
    res = dr_aci(oracle_scores, true_residuals,
                 alpha=alpha, gamma=gamma, n_warmup=n_warmup)
    results['oracle'] = (res.coverage, res.avg_width)

    return results


# ---------------------------------------------------------------------------
# Full MC study
# ---------------------------------------------------------------------------
def run_simulation(cfg: dict, nuisance_method: str = 'xgboost') -> dict:
    """Run full Monte Carlo study over rho x T grid.

    Uses joblib for parallel MC trials when cfg['n_jobs'] > 1.
    Deterministic regardless of n_jobs (each trial has its own seed).
    """
    from joblib import Parallel, delayed

    methods = cfg['methods']
    rhos = cfg['rhos']
    sample_sizes = cfg['sample_sizes']
    n_mc = cfg['n_mc']
    n_jobs = cfg.get('n_jobs', 1)
    base_seed = cfg['seed']

    results = {}
    total_configs = len(rhos) * len(sample_sizes)
    config_idx = 0
    t0 = time.time()

    for rho in rhos:
        for T in sample_sizes:
            config_idx += 1
            key = (rho, T)
            results[key] = {m: [] for m in methods}

            # Deterministic per-trial seeds
            seeds = [base_seed + config_idx * 100_000 + mc
                     for mc in range(n_mc)]

            if n_jobs == 1:
                # Sequential: show progress
                trial_results = []
                for mc, seed in enumerate(seeds):
                    trial = run_one_trial(T, rho, seed, cfg, nuisance_method)
                    trial_results.append(trial)
                    if (mc + 1) % max(n_mc // 10, 1) == 0:
                        elapsed = time.time() - t0
                        print(f"  [{config_idx}/{total_configs}] rho={rho}, "
                              f"T={T}: {mc+1}/{n_mc} done  "
                              f"({elapsed:.0f}s elapsed)")
            else:
                # Parallel via joblib
                print(f"  [{config_idx}/{total_configs}] rho={rho}, "
                      f"T={T}: dispatching {n_mc} trials on {n_jobs} workers")
                trial_results = Parallel(n_jobs=n_jobs)(
                    delayed(run_one_trial)(T, rho, seed, cfg, nuisance_method)
                    for seed in seeds
                )
                elapsed = time.time() - t0
                print(f"  [{config_idx}/{total_configs}] rho={rho}, "
                      f"T={T}: {n_mc}/{n_mc} done  ({elapsed:.0f}s elapsed)")

            for trial in trial_results:
                for m in methods:
                    results[key][m].append(trial[m])

    print(f"Total simulation time: {time.time() - t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Save raw results to CSV
# ---------------------------------------------------------------------------
def save_raw_results(results: dict, cfg: dict):
    """Save raw MC results to CSV for reproducibility."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS_DIR / 'sim_raw.csv'

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rho', 'T', 'method', 'mc_rep', 'coverage', 'width'])
        for (rho, T), method_results in results.items():
            for method, trials in method_results.items():
                for mc_idx, (cov, width) in enumerate(trials):
                    writer.writerow([rho, T, method, mc_idx, f'{cov:.6f}',
                                     f'{width:.6f}'])
    print(f"Saved raw results: {outpath}")


# ---------------------------------------------------------------------------
# Figure 1: coverage vs mixing strength
# ---------------------------------------------------------------------------
def make_figure(results: dict, cfg: dict):
    """Figure 1: Coverage (top) and PIAW (bottom) vs rho for each T.

    Two-row layout:
      Top row:    Empirical coverage with full y-axis [0, 1.05] so HAC
                  collapse to ~0.19 is visible.
      Bottom row: Prediction interval average width — where methods
                  clearly separate (DR-ACI ~8.6, VS-DR-ACI ~3.3, HAC ~0.4).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    methods = cfg['methods']
    sample_sizes = cfg['sample_sizes']
    rhos = cfg['rhos']
    alpha = cfg['alpha']

    colors = METHOD_COLORS
    markers = METHOD_MARKERS

    n_T = len(sample_sizes)
    fig, axes = plt.subplots(2, n_T,
                             figsize=(4.5 * n_T, 7),
                             sharex='col')
    if n_T == 1:
        axes = axes.reshape(2, 1)

    for col, T in enumerate(sample_sizes):
        ax_cov = axes[0, col]
        ax_width = axes[1, col]

        for m in methods:
            coverages, cov_ses = [], []
            widths, width_ses = [], []
            for rho in rhos:
                cov_width = results[(rho, T)][m]
                covs = [c for c, w in cov_width]
                ws = [w for c, w in cov_width]
                coverages.append(np.mean(covs))
                cov_ses.append(np.std(covs) / np.sqrt(len(covs)))
                widths.append(np.mean(ws))
                width_ses.append(np.std(ws) / np.sqrt(len(ws)))

            coverages = np.array(coverages)
            cov_ses = np.array(cov_ses)
            widths = np.array(widths)
            width_ses = np.array(width_ses)

            lbl = METHOD_LABELS[m]

            # Coverage panel
            ax_cov.plot(rhos, coverages, marker=markers[m], color=colors[m],
                        label=lbl, linewidth=1.2, markersize=4)
            ax_cov.fill_between(rhos, coverages - 2 * cov_ses,
                                coverages + 2 * cov_ses,
                                color=colors[m], alpha=0.08)

            # Width panel
            ax_width.plot(rhos, widths, marker=markers[m], color=colors[m],
                          label=lbl, linewidth=1.2, markersize=4)
            ax_width.fill_between(rhos, widths - 2 * width_ses,
                                  widths + 2 * width_ses,
                                  color=colors[m], alpha=0.08)

        # Coverage panel styling
        ax_cov.axhline(1 - alpha, color='black', linestyle='--',
                       linewidth=0.8, label=f'Nominal ({1 - alpha:.0%})')
        ax_cov.set_title(f'$T = {T}$')
        ax_cov.set_ylim(0.0, 1.05)
        ax_cov.set_xticks(rhos)
        ax_cov.grid(True, alpha=0.3)

        # Width panel styling
        ax_width.set_xlabel(r'AR(1) coefficient $\rho$')
        ax_width.set_xticks(rhos)
        ax_width.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Empirical coverage')
    axes[1, 0].set_ylabel('PIAW')
    # Single legend in bottom-right panel
    axes[1, -1].legend(fontsize=6, loc='upper right', ncol=2)
    fig.tight_layout()
    outpath = FIG_DIR / 'coverage_vs_mixing.pdf'
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    print(f"Saved figure: {outpath}")
    plt.close()


# ---------------------------------------------------------------------------
# Table 1: 9 methods x 3 rho values (paper format)
# ---------------------------------------------------------------------------
def make_table(results: dict, cfg: dict):
    """Table 1: Coverage and PIAW for 9 methods at rho={0, 0.6, 0.9}, T=2000.

    Format: methods as rows, (rho, metric) as columns.
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Use largest T available
    T_table = max(cfg['sample_sizes'])
    methods = cfg['methods']
    n_mc = cfg['n_mc']

    # Select the 3 representative rhos that exist in results
    table_rhos = [r for r in TABLE_RHOS if (r, T_table) in results]
    if not table_rhos:
        table_rhos = cfg['rhos'][:3]

    n_rho = len(table_rhos)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Empirical coverage and prediction interval average width (PIAW) "
        r"across mixing strengths ($T=" + str(T_table) +
        r"$, $\alpha=" + str(cfg['alpha']) + r"$, " +
        str(n_mc) + r" replications). "
        r"Standard errors in parentheses.}"
    )
    lines.append(r"\label{tab:coverage}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cc" * n_rho + "}")
    lines.append(r"\toprule")

    # Header: rho values spanning 2 columns each
    header = "Method"
    for rho in table_rhos:
        header += r" & \multicolumn{2}{c}{$\rho = " + f"{rho}$" + "}"
    header += r" \\"
    lines.append(header)

    # Sub-header
    subheader = ""
    for _ in table_rhos:
        subheader += r" & Cov. & PIAW"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Data rows
    for m in methods:
        label = METHOD_LABELS[m].replace('--', r'--')
        row = label
        for rho in table_rhos:
            cov_width = results[(rho, T_table)][m]
            covs = [c for c, w in cov_width]
            widths = [w for c, w in cov_width]
            avg_cov = np.mean(covs)
            se_cov = np.std(covs) / np.sqrt(len(covs))
            avg_width = np.mean(widths)
            se_width = np.std(widths) / np.sqrt(len(widths))
            row += (f" & {avg_cov:.2f}"
                    r"\tiny{" + f"({se_cov:.3f})" + "}"
                    f" & {avg_width:.2f}"
                    r"\tiny{" + f"({se_width:.2f})" + "}")
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    outpath = TAB_DIR / "coverage_table.tex"
    outpath.write_text(tex_content)
    print(f"Saved table: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_sim_args()
    cfg = get_config(quick=args.quick, n_jobs=args.n_jobs)

    if args.n_mc is not None:
        cfg['n_mc'] = args.n_mc

    nuisance_method = 'linear' if args.linear else 'xgboost'

    print("=" * 60)
    print("DR-ACI Coverage Simulation (Section 5)")
    print("=" * 60)
    print(f"Config: rhos={cfg['rhos']}, T={cfg['sample_sizes']}, "
          f"N_MC={cfg['n_mc']}")
    print(f"alpha={cfg['alpha']}, gamma={cfg['gamma']}, seed={cfg['seed']}")
    print(f"Nuisance: {nuisance_method}")
    print(f"Methods: {cfg['methods']}")
    print(f"n_jobs: {cfg['n_jobs']}")
    print(f"Block partition: train={cfg['train_frac']:.0%}, "
          f"gap={cfg['gap_frac']:.0%}, "
          f"cal={1 - cfg['train_frac'] - cfg['gap_frac']:.0%}")
    print()

    results = run_simulation(cfg, nuisance_method)
    save_raw_results(results, cfg)
    make_figure(results, cfg)
    make_table(results, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
