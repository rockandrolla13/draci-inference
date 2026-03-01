"""
CLI entry point for DR-ACI empirical experiments on Dynamic M-ELO.

Usage:
    python run_all.py --experiment cs        # cross-sectional only
    python run_all.py --experiment panel      # panel DR-ACI only
    python run_all.py --experiment mixing     # mixing diagnostics only
    python run_all.py --experiment sensitivity # sensitivity analysis
    python run_all.py --experiment misspec    # misspecification robustness
    python run_all.py --experiment all        # run everything
    python run_all.py --experiment all --dev  # quick dev run (20% subsample)
"""

import time
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='DR-ACI Empirical Experiments on Dynamic M-ELO'
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['cs', 'panel', 'mixing', 'sensitivity', 'misspec', 'all'],
        default='all',
        help='Which experiment to run',
    )
    parser.add_argument(
        '--dev', action='store_true',
        help='Use 20%% subsample for faster development runs',
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full production run (no subsampling)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = args.dev and not args.full

    t0 = time.time()

    print("=" * 70)
    print("DR-ACI Empirical Experiments on Dynamic M-ELO Data")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Mode: {'dev (20%% subsample)' if dev else 'production (full data)'}")
    print()

    # Pre-load data once for experiments that need it
    panel_df = None
    if args.experiment in ['panel', 'mixing', 'sensitivity', 'misspec', 'all']:
        from empirical.data_prep import prepare_panel
        panel_df = prepare_panel(dev=dev)

    # --- Experiment 2a: Cross-sectional ---
    if args.experiment in ['cs', 'all']:
        print("\n" + "=" * 60)
        print("Experiment 2a: Cross-Sectional DR-ACI")
        print("=" * 60)
        from empirical.cross_sectional import run_cross_sectional, save_cs_results
        results, att_check, meta = run_cross_sectional()
        save_cs_results(results, att_check, meta)

    # --- Experiment 2b: Panel ---
    if args.experiment in ['panel', 'all']:
        print("\n" + "=" * 60)
        print("Experiment 2b: Panel DR-ACI")
        print("=" * 60)
        from empirical.panel_draci import run_panel_experiment, save_panel_results
        method_results, seq_df = run_panel_experiment(panel_df, dev=dev)
        save_panel_results(method_results, seq_df)

    # --- Experiment 3: Mixing diagnostics ---
    if args.experiment in ['mixing', 'all']:
        print("\n" + "=" * 60)
        print("Experiment 3: Mixing Diagnostics")
        print("=" * 60)
        from empirical.mixing_diagnostics import run_mixing_diagnostics
        run_mixing_diagnostics(panel_df)

    # --- Experiment 4: Sensitivity ---
    if args.experiment in ['sensitivity', 'all']:
        print("\n" + "=" * 60)
        print("Experiment 4: Sensitivity Analysis (Appendix)")
        print("=" * 60)
        from empirical.sensitivity import run_experiment_4
        run_experiment_4(dev=dev)

    # --- Experiment 5: Misspecification ---
    if args.experiment in ['misspec', 'all']:
        print("\n" + "=" * 60)
        print("Experiment 5: Misspecification Robustness (Appendix)")
        print("=" * 60)
        from empirical.misspecification import run_experiment_5
        run_experiment_5(dev=dev)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
