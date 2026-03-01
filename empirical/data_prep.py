"""
Data preparation for DR-ACI empirical application on Dynamic M-ELO.

Loads the existing processed parquet data, constructs treatment/outcome/covariate
variables, and provides both cross-sectional and panel data frames.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from empirical.config import (
    DAILY_SEC_FILE, ADOPTION_FILE,
    CS_CUTOFF_DATE, CS_POST_START, CS_POST_END, CS_PRE_START, CS_PRE_END,
    PANEL_START, PANEL_END, OUTCOME_COL, COVARIATES, DATA_COLS,
    DEV_SUBSAMPLE_FRAC, DEV_TICKERS,
)


def load_daily_data(columns: list[str] | None = None) -> pd.DataFrame:
    """Load daily SEC MIDAS data, aggregate to ticker-day level.

    The raw data is at exchange-ticker-day granularity (62M rows).
    We aggregate to ticker-day by summing volumes and taking first-of-day
    for rank covariates.
    """
    if columns is None:
        columns = DATA_COLS

    print(f"Loading {DAILY_SEC_FILE}...")
    df = pd.read_parquet(DAILY_SEC_FILE, columns=columns)
    print(f"  Raw rows: {len(df):,}")

    # Aggregate to ticker-day (sum volumes, mean ranks)
    vol_cols = [c for c in ['hidden_vol', 'trade_vol', 'trade_vol_for_hidden',
                            'hidden', 'trades'] if c in df.columns]
    rank_cols = [c for c in ['mcap_rank', 'turn_rank', 'volatility_rank',
                             'price_rank'] if c in df.columns]

    agg_dict = {}
    for c in vol_cols:
        agg_dict[c] = 'sum'
    for c in rank_cols:
        agg_dict[c] = 'first'
    for c in ['adoption_date', 'adoption_method', 'vix']:
        if c in df.columns:
            agg_dict[c] = 'first'

    df_td = df.groupby(['ticker', 'date'], as_index=False).agg(agg_dict)

    # Compute hidden_share at ticker-day level
    if 'trade_vol_for_hidden' in df_td.columns and 'hidden_vol' in df_td.columns:
        denom = df_td['trade_vol_for_hidden'].replace(0, np.nan)
        df_td['hidden_share'] = df_td['hidden_vol'] / denom
    elif 'hidden' in df_td.columns and 'trades' in df_td.columns:
        denom = df_td['trades'].replace(0, np.nan)
        df_td['hidden_share'] = df_td['hidden'] / denom

    print(f"  Ticker-day rows: {len(df_td):,}")
    print(f"  Unique tickers: {df_td['ticker'].nunique():,}")
    print(f"  Date range: {df_td['date'].min()} to {df_td['date'].max()}")

    return df_td


def prepare_cross_sectional(df_td: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prepare cross-sectional dataset (Experiment 2a).

    Design:
      - W_i = 1 if adoption_date <= cutoff (2024-05-10), else 0
      - Y_i = mean(hidden_share) over post-cutoff window
      - X_i = pre-treatment means of covariates

    Returns one row per ticker with columns:
      ticker, W, Y, mcap_rank, turn_rank, volatility_rank, price_rank,
      pre_hidden_share, adoption_date
    """
    if df_td is None:
        df_td = load_daily_data()

    cutoff = pd.Timestamp(CS_CUTOFF_DATE)
    pre_start = pd.Timestamp(CS_PRE_START)
    pre_end = pd.Timestamp(CS_PRE_END)
    post_start = pd.Timestamp(CS_POST_START)
    post_end = pd.Timestamp(CS_POST_END)

    # Ensure date is datetime
    df_td['date'] = pd.to_datetime(df_td['date'])
    if 'adoption_date' in df_td.columns:
        df_td['adoption_date'] = pd.to_datetime(df_td['adoption_date'])

    # Treatment: adopted by cutoff
    ticker_adoption = df_td.groupby('ticker')['adoption_date'].first().reset_index()
    ticker_adoption['W'] = (ticker_adoption['adoption_date'] <= cutoff).astype(int)
    # Tickers without adoption date are control
    ticker_adoption.loc[ticker_adoption['adoption_date'].isna(), 'W'] = 0

    # Post-treatment outcome: mean hidden_share
    post_mask = (df_td['date'] >= post_start) & (df_td['date'] <= post_end)
    y_post = (df_td.loc[post_mask]
              .groupby('ticker')['hidden_share']
              .mean()
              .reset_index()
              .rename(columns={'hidden_share': 'Y'}))

    # Pre-treatment covariates: means
    pre_mask = (df_td['date'] >= pre_start) & (df_td['date'] <= pre_end)
    rank_cols = ['mcap_rank', 'turn_rank', 'volatility_rank', 'price_rank']
    available_ranks = [c for c in rank_cols if c in df_td.columns]

    x_pre = df_td.loc[pre_mask].groupby('ticker').agg(
        **{c: (c, 'mean') for c in available_ranks},
        pre_hidden_share=('hidden_share', 'mean'),
    ).reset_index()

    # Merge
    cs = (ticker_adoption
          .merge(y_post, on='ticker', how='inner')
          .merge(x_pre, on='ticker', how='inner'))

    # Drop rows with missing outcome or covariates
    cs = cs.dropna(subset=['Y'] + available_ranks + ['pre_hidden_share'])

    n_treated = cs['W'].sum()
    n_control = len(cs) - n_treated
    print(f"Cross-sectional: {len(cs)} tickers "
          f"({n_treated} treated, {n_control} control)")

    return cs


def prepare_panel(df_td: pd.DataFrame | None = None,
                  dev: bool = False) -> pd.DataFrame:
    """Prepare panel dataset (Experiment 2b).

    Returns ticker-day dataframe with columns:
      ticker, date, W (treatment indicator), Y (hidden_share),
      covariates (pre-treatment means), adoption_date

    If dev=True, returns a 20% stratified subsample by ticker.
    """
    if df_td is None:
        df_td = load_daily_data()

    panel_start = pd.Timestamp(PANEL_START)
    panel_end = pd.Timestamp(PANEL_END)

    df_td['date'] = pd.to_datetime(df_td['date'])
    if 'adoption_date' in df_td.columns:
        df_td['adoption_date'] = pd.to_datetime(df_td['adoption_date'])

    # Filter to panel window
    mask = (df_td['date'] >= panel_start) & (df_td['date'] <= panel_end)
    panel = df_td.loc[mask].copy()

    # Treatment indicator: W_it = 1{date >= adoption_date_i}
    panel['W'] = 0
    has_adoption = panel['adoption_date'].notna()
    panel.loc[has_adoption, 'W'] = (
        panel.loc[has_adoption, 'date'] >= panel.loc[has_adoption, 'adoption_date']
    ).astype(int)

    panel['Y'] = panel['hidden_share']

    # Pre-treatment covariates (ticker-level means from pre-period)
    cutoff = pd.Timestamp(CS_CUTOFF_DATE)
    pre_mask = panel['date'] < cutoff
    rank_cols = ['mcap_rank', 'turn_rank', 'volatility_rank', 'price_rank']
    available_ranks = [c for c in rank_cols if c in panel.columns]

    pre_means = panel.loc[pre_mask].groupby('ticker').agg(
        **{c: (c, 'mean') for c in available_ranks},
        pre_hidden_share=('hidden_share', 'mean'),
    ).reset_index()

    # Merge covariates back (constant per ticker)
    panel = panel.merge(pre_means, on='ticker', how='left',
                        suffixes=('', '_pre'))

    # Drop tickers with missing covariates
    covariate_cols = [c + '_pre' if c + '_pre' in panel.columns else c
                      for c in available_ranks] + ['pre_hidden_share']
    # Resolve actual column names
    actual_cov_cols = []
    for c in available_ranks:
        if c + '_pre' in panel.columns:
            actual_cov_cols.append(c + '_pre')
        elif c in panel.columns:
            actual_cov_cols.append(c)
    actual_cov_cols.append('pre_hidden_share')

    panel = panel.dropna(subset=['Y'] + actual_cov_cols)

    if dev:
        # 20% stratified subsample by ticker
        tickers = panel['ticker'].unique()
        rng = np.random.default_rng(42)
        n_sample = min(DEV_TICKERS, int(len(tickers) * DEV_SUBSAMPLE_FRAC))
        sample_tickers = rng.choice(tickers, size=n_sample, replace=False)
        panel = panel[panel['ticker'].isin(sample_tickers)]
        print(f"Dev subsample: {len(sample_tickers)} tickers, "
              f"{len(panel):,} rows")

    print(f"Panel: {panel['ticker'].nunique()} tickers, "
          f"{len(panel):,} rows, "
          f"date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    return panel


def get_covariate_matrix(df: pd.DataFrame, covariates: list[str] | None = None
                         ) -> np.ndarray:
    """Extract covariate matrix X from dataframe.

    Handles column name resolution (with _pre suffix from panel merge).
    """
    if covariates is None:
        covariates = COVARIATES

    cols = []
    for c in covariates:
        if c in df.columns:
            cols.append(c)
        elif c + '_pre' in df.columns:
            cols.append(c + '_pre')
        else:
            raise KeyError(f"Covariate '{c}' not found in dataframe. "
                           f"Available: {list(df.columns)}")

    X = df[cols].values.astype(np.float64)
    # Impute any remaining NaN with column median
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X
