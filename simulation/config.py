"""
Configuration for DR-ACI simulation study (Section 5 of the paper).

Central configuration for simulation parameters and output paths.
Supports --quick mode for development and --full for production runs.

Method metadata (labels, colors, markers) lives in draci.methods.
DGP mathematical constants live in draci.dgp.
"""

from pathlib import Path
import argparse

from draci.methods import METHODS, METHOD_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SIM_DIR.parent
PAPER_DIR = PROJECT_DIR / "paper"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Simulation grid parameters (Section 5 of paper)
# ---------------------------------------------------------------------------
RHOS = [0.0, 0.3, 0.6, 0.79, 0.9, 0.95]  # AR(1) coefficients; 0.79 calibrated to M-ELO
SAMPLE_SIZES = [500, 1000, 2000]

# ---------------------------------------------------------------------------
# Estimation parameters
# ---------------------------------------------------------------------------
N_MC = 500             # Monte Carlo repetitions (paper value)
ALPHA = 0.10           # target miscoverage rate (90% coverage)
GAMMA = 0.005          # ACI step size (paper value; Lemma 3)
SEED = 42
TRAIN_FRAC = 0.40      # training block fraction
GAP_FRAC = 0.10        # decorrelation gap fraction
                        # calibration = 1 - TRAIN_FRAC - GAP_FRAC = 0.50
N_WARMUP_FRAC = 0.10   # warmup fraction of calibration set

# Block CP parameters
BLOCK_SIZE = 10         # block size for Block CP method

# NexCP parameters
NEXCP_LAMBDA = 0.05    # exponential decay rate

# Table 1 uses these 3 representative rho values
TABLE_RHOS = [0.0, 0.6, 0.9]

# ---------------------------------------------------------------------------
# Quick mode overrides (for development/testing)
# ---------------------------------------------------------------------------
QUICK_RHOS = [0.0, 0.9]
QUICK_SAMPLE_SIZES = [500]
QUICK_N_MC = 10


def get_config(quick: bool = False, n_jobs: int = 1) -> dict:
    """Return configuration dict, optionally in quick mode.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers for joblib (1 = sequential).
    """
    if quick:
        return {
            'rhos': QUICK_RHOS,
            'sample_sizes': QUICK_SAMPLE_SIZES,
            'n_mc': QUICK_N_MC,
            'alpha': ALPHA,
            'gamma': GAMMA,
            'seed': SEED,
            'train_frac': TRAIN_FRAC,
            'gap_frac': GAP_FRAC,
            'methods': METHODS,
            'n_jobs': n_jobs,
        }
    return {
        'rhos': RHOS,
        'sample_sizes': SAMPLE_SIZES,
        'n_mc': N_MC,
        'alpha': ALPHA,
        'gamma': GAMMA,
        'seed': SEED,
        'train_frac': TRAIN_FRAC,
        'gap_frac': GAP_FRAC,
        'methods': METHODS,
        'n_jobs': n_jobs,
    }


def parse_sim_args() -> argparse.Namespace:
    """Parse command-line arguments for simulation runner."""
    parser = argparse.ArgumentParser(description='DR-ACI Coverage Simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 10 MC, 2 rho, 1 T')
    parser.add_argument('--n-mc', type=int, default=None,
                        help='Override number of MC repetitions')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--linear', action='store_true',
                        help='Use linear nuisance estimators (fast)')
    return parser.parse_args()
