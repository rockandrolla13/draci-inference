"""
Configuration for DR-ACI empirical application on Dynamic M-ELO data.

The M-ELO dataset (62M rows, ~9,700 tickers, daily, Jan 2023 - Jun 2025)
provides ideal conditions for testing DR-ACI:
  - Strong serial dependence (rho ~ 0.79)
  - Staggered treatment with known assignment dates
  - Rich pre-treatment covariates
"""

import os
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMPIRICAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EMPIRICAL_DIR.parent
PAPER_DIR = PROJECT_DIR / "paper"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"
RESULTS_DIR = EMPIRICAL_DIR / "results"

# M-ELO data location (set MELO_DATA_DIR env var; no hardcoded fallback)
_melo_dir = os.environ.get("MELO_DATA_DIR")
MELO_DATA_DIR = Path(_melo_dir) if _melo_dir else None

def get_melo_data_dir() -> Path:
    """Return M-ELO data directory, or raise with helpful message."""
    if MELO_DATA_DIR is None:
        raise EnvironmentError(
            "MELO_DATA_DIR environment variable not set. "
            "Set it to the path containing processed M-ELO data, e.g.:\n"
            "  export MELO_DATA_DIR=/path/to/dynamic-melo/data/processed"
        )
    return MELO_DATA_DIR

DAILY_SEC_FILE = (MELO_DATA_DIR / "analysis_daily_sec.parquet") if MELO_DATA_DIR else None
ADOPTION_FILE = (MELO_DATA_DIR / "adoption_dates.parquet") if MELO_DATA_DIR else None

# ---------------------------------------------------------------------------
# Treatment design
# ---------------------------------------------------------------------------
# Cutoff for cross-sectional design (midpoint of staggered rollout)
CS_CUTOFF_DATE = date(2024, 5, 10)

# Post-treatment window for cross-sectional outcome
CS_POST_START = date(2024, 5, 10)
CS_POST_END = date(2024, 6, 30)

# Pre-treatment window for covariates
CS_PRE_START = date(2024, 1, 1)
CS_PRE_END = date(2024, 4, 14)

# Panel date range
PANEL_START = date(2023, 6, 1)
PANEL_END = date(2025, 3, 31)

# Adoption cohorts (from dynamic-melo project)
COHORT_DATES = {
    "test": date(2024, 4, 15),
    "WZ": date(2024, 5, 6),
    "TV": date(2024, 5, 8),
    "MS": date(2024, 5, 13),
    "AL": date(2024, 5, 15),
}

# ---------------------------------------------------------------------------
# Outcome and covariate definitions
# ---------------------------------------------------------------------------
OUTCOME_COL = "hidden_share"  # hidden_vol / trade_vol

# Pre-treatment covariates (ticker-level means)
COVARIATES = [
    "mcap_rank",
    "turn_rank",
    "volatility_rank",
    "price_rank",
    "pre_hidden_share",  # constructed: pre-treatment mean of hidden_share
]

# Additional columns needed from the data
DATA_COLS = [
    "date", "ticker", "adoption_date", "adoption_method",
    "mcap_rank", "turn_rank", "volatility_rank", "price_rank",
    "hidden", "hidden_vol", "trade_vol", "trade_vol_for_hidden",
    "trades", "vix",
]

# ---------------------------------------------------------------------------
# Estimation parameters
# ---------------------------------------------------------------------------
ALPHA = 0.10           # target miscoverage (90% coverage)
GAMMA = 0.005          # ACI step size
K_BLOCKS = 5           # temporal cross-fitting blocks
CLIP_PROPENSITY = (0.05, 0.95)

# TWFE ATT estimate (from existing econometrics pipeline)
TWFE_ATT = 0.036

# ---------------------------------------------------------------------------
# Dev/quick mode
# ---------------------------------------------------------------------------
DEV_SUBSAMPLE_FRAC = 0.20  # 20% stratified subsample for development
DEV_TICKERS = 500           # max tickers in dev mode

# ---------------------------------------------------------------------------
# Sensitivity analysis ranges (Experiment 4)
# ---------------------------------------------------------------------------
SENSITIVITY_GAMMAS = [0.001, 0.005, 0.01, 0.02, 0.05]
SENSITIVITY_K = [3, 5, 7, 10]
SENSITIVITY_GAP_FRAC = [0.0, 0.05, 0.10]

# ---------------------------------------------------------------------------
# Mixing diagnostics (Experiment 3)
# ---------------------------------------------------------------------------
MIXING_MAX_LAG = 100  # max lag for ACF
MIXING_BETA_LAGS = [1, 2, 5, 10, 20, 50, 100]  # lags for beta-mixing estimation
MIXING_N_BINS = 20    # histogram bins for TV distance estimation
