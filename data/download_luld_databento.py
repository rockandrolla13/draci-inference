#!/usr/bin/env python3
"""
Download LULD limit state and halt data from Databento for CL-F1.

Strategy:
  1. Status schema  → W=1 events (LULD halts with microsecond timestamps)
  2. Statistics schema → LULD bands (upper/lower price limits) throughout the day
  3. BBO-1s schema  → 1-second best bid/offer to detect W=0 limit state entries
  4. OHLCV-1m schema → 1-min bars around halt events for outcome construction

Dataset: DBEQ.BASIC (IEX + NYSE Chicago/National + MIAX Pearl)
  - Free to redistribute, historical from Apr 2023
  - Status schema: $4/GB
  - Statistics schema: $4/GB
  - BBO-1s: $4/GB
  - OHLCV-1m: $4/GB

Usage:
  export DATABENTO_API_KEY=db-XXXXXXXX
  python download_luld_databento.py --step cost     # estimate cost first
  python download_luld_databento.py --step status    # download W=1 halt events
  python download_luld_databento.py --step stats     # download LULD bands
  python download_luld_databento.py --step bbo       # download BBO for W=0 detection
  python download_luld_databento.py --step ohlcv     # download 1-min bars around halts
  python download_luld_databento.py --step all       # run all steps
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import databento as db
except ImportError:
    print("Install databento: pip install databento")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
HALTS_DIR = RAW_DIR / "nyse_halts"
DATABENTO_DIR = RAW_DIR / "databento"
DATABENTO_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "DBEQ.BASIC"
# DBEQ.BASIC historical starts Apr 2023
START = "2023-04-01"
END = "2025-12-31"

# We focus on symbols that had LULD halts (from NYSE halt file)
LULD_CONSOLIDATED = HALTS_DIR / "luld_events_2020_2025.csv"


def load_luld_symbols(start_date: str = "2023-04-01") -> dict:
    """Load LULD-halted symbols and their halt dates from NYSE data.

    Returns dict: {symbol: [list of halt dates as strings]}
    """
    symbols = {}
    with open(LULD_CONSOLIDATED) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Halt Date"] >= start_date:
                sym = row["Symbol"].strip()
                symbols.setdefault(sym, []).append(row["Halt Date"])
    return symbols


def get_client() -> db.Historical:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("ERROR: Set DATABENTO_API_KEY environment variable")
        print("Sign up at https://databento.com/signup for $125 free credit")
        sys.exit(1)
    return db.Historical(key)


# ---------------------------------------------------------------------------
# Step 1: Cost estimation
# ---------------------------------------------------------------------------
def estimate_costs(client: db.Historical):
    """Estimate cost of each download before committing."""
    print("=" * 60)
    print("COST ESTIMATION (DBEQ.BASIC)")
    print("=" * 60)

    schemas = [
        ("status", "W=1 halt events"),
        ("statistics", "LULD bands (upper/lower price limits)"),
        ("bbo-1s", "1-second BBO for W=0 limit state detection"),
        ("ohlcv-1m", "1-minute OHLCV for outcome construction"),
    ]

    total = 0.0
    for schema, desc in schemas:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                schema=schema,
                start=START,
                end=END,
            )
            print(f"\n  {schema:12s} — {desc}")
            print(f"    Cost: ${cost:.2f}")
            total += cost
        except Exception as e:
            print(f"\n  {schema:12s} — {desc}")
            print(f"    Error: {e}")

    print(f"\n{'='*60}")
    print(f"  TOTAL ESTIMATED: ${total:.2f}")
    print(f"  Available credit: $125.00")
    print(f"  Remaining after: ${125 - total:.2f}")
    print(f"{'='*60}")

    # Also estimate per-symbol costs for targeted download
    luld_syms = load_luld_symbols()
    top_syms = sorted(luld_syms.keys(), key=lambda s: len(luld_syms[s]), reverse=True)[:50]
    print(f"\nTop 50 LULD symbols (by halt frequency):")
    for sym in top_syms[:10]:
        print(f"  {sym}: {len(luld_syms[sym])} halts")
    print(f"  ... ({len(top_syms)} symbols)")

    # Estimate cost for targeted BBO download (top 50 symbols only)
    try:
        cost_targeted = client.metadata.get_cost(
            dataset=DATASET,
            symbols=top_syms,
            stype_in="raw_symbol",
            schema="bbo-1s",
            start=START,
            end=END,
        )
        print(f"\n  Targeted BBO (top 50 symbols): ${cost_targeted:.2f}")
    except Exception as e:
        print(f"\n  Targeted BBO cost error: {e}")


# ---------------------------------------------------------------------------
# Step 2: Download status schema (W=1 events)
# ---------------------------------------------------------------------------
def download_status(client: db.Historical):
    """Download all LULD halt/resume status messages."""
    print("\n[STATUS] Downloading halt/resume events...")
    outfile = DATABENTO_DIR / "dbeq_status_all.dbn.zst"

    if outfile.exists():
        print(f"  Already exists: {outfile}")
        return outfile

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols="ALL_SYMBOLS",
        stype_in="raw_symbol",
        schema="status",
        start=START,
        end=END,
    )
    data.to_file(str(outfile))
    print(f"  Saved: {outfile}")

    # Convert to CSV for inspection
    df = data.to_df()
    csv_out = DATABENTO_DIR / "dbeq_status_all.csv"
    df.to_csv(csv_out)
    print(f"  CSV: {csv_out} ({len(df)} records)")

    # Filter LULD events
    luld_mask = df["reason"].astype(str).str.contains("LULD", case=False, na=False)
    luld_df = df[luld_mask]
    luld_csv = DATABENTO_DIR / "dbeq_status_luld.csv"
    luld_df.to_csv(luld_csv)
    print(f"  LULD events: {luld_csv} ({len(luld_df)} records)")

    return outfile


# ---------------------------------------------------------------------------
# Step 3: Download statistics schema (LULD bands)
# ---------------------------------------------------------------------------
def download_statistics(client: db.Historical):
    """Download LULD band prices (upper/lower limits)."""
    print("\n[STATISTICS] Downloading LULD bands...")

    # Get symbols that had LULD halts
    luld_syms = load_luld_symbols()
    # Take all unique symbols (or top N if too many)
    all_syms = list(luld_syms.keys())
    print(f"  {len(all_syms)} symbols with LULD halts since {START}")

    # Download in batches to manage cost
    # Start with top 200 most-halted symbols
    top_syms = sorted(all_syms, key=lambda s: len(luld_syms[s]), reverse=True)[:200]
    outfile = DATABENTO_DIR / "dbeq_statistics_luld_top200.dbn.zst"

    if outfile.exists():
        print(f"  Already exists: {outfile}")
        return outfile

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=top_syms,
        stype_in="raw_symbol",
        schema="statistics",
        start=START,
        end=END,
    )
    data.to_file(str(outfile))
    print(f"  Saved: {outfile}")

    df = data.to_df()
    csv_out = DATABENTO_DIR / "dbeq_statistics_luld_top200.csv"
    df.to_csv(csv_out)
    print(f"  CSV: {csv_out} ({len(df)} records)")

    return outfile


# ---------------------------------------------------------------------------
# Step 4: Download BBO for W=0 limit state detection
# ---------------------------------------------------------------------------
def download_bbo(client: db.Historical):
    """Download 1-second BBO for W=0 limit state detection.

    Strategy: For each LULD-halted symbol on each halt date,
    download the full day's BBO. W=0 events are identified as
    timestamps where NBBO = LULD band for < 15 seconds without
    a subsequent halt.
    """
    print("\n[BBO] Downloading 1-second BBO for limit state detection...")

    luld_syms = load_luld_symbols()

    # Group by date to minimize API calls
    date_symbols = {}
    for sym, dates in luld_syms.items():
        for d in dates:
            if d >= START:
                date_symbols.setdefault(d, set()).add(sym)

    print(f"  {len(date_symbols)} unique halt dates")
    print(f"  Downloading BBO for halt dates (top 50 symbols)...")

    # For cost management: start with top 50 most-halted symbols
    top_syms = sorted(luld_syms.keys(), key=lambda s: len(luld_syms[s]), reverse=True)[:50]

    outfile = DATABENTO_DIR / "dbeq_bbo1s_luld_top50.dbn.zst"
    if outfile.exists():
        print(f"  Already exists: {outfile}")
        return outfile

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=top_syms,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=START,
        end=END,
    )
    data.to_file(str(outfile))
    print(f"  Saved: {outfile}")

    return outfile


# ---------------------------------------------------------------------------
# Step 5: Download OHLCV-1m around halt events
# ---------------------------------------------------------------------------
def download_ohlcv(client: db.Historical):
    """Download 1-minute OHLCV for outcome construction around halts."""
    print("\n[OHLCV] Downloading 1-minute bars for outcome construction...")

    luld_syms = load_luld_symbols()
    # All symbols that had LULD halts
    all_syms = list(luld_syms.keys())
    print(f"  {len(all_syms)} symbols")

    outfile = DATABENTO_DIR / "dbeq_ohlcv1m_luld_all.dbn.zst"
    if outfile.exists():
        print(f"  Already exists: {outfile}")
        return outfile

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=all_syms,
        stype_in="raw_symbol",
        schema="ohlcv-1m",
        start=START,
        end=END,
    )
    data.to_file(str(outfile))
    print(f"  Saved: {outfile}")

    df = data.to_df()
    csv_out = DATABENTO_DIR / "dbeq_ohlcv1m_luld_all.csv"
    df.to_csv(csv_out)
    print(f"  CSV: {csv_out} ({len(df)} records)")

    return outfile


# ---------------------------------------------------------------------------
# Step 6: Identify W=0 events from BBO + bands
# ---------------------------------------------------------------------------
def identify_w0_events():
    """Post-processing: identify W=0 (limit state resolved) events.

    W=0: NBBO touched LULD band for < 15 seconds, no halt followed.
    Requires: BBO data + LULD band data + status data (halts).
    """
    print("\n[W=0 IDENTIFICATION] Reconstructing limit state resolutions...")

    bbo_file = DATABENTO_DIR / "dbeq_bbo1s_luld_top50.dbn.zst"
    stats_file = DATABENTO_DIR / "dbeq_statistics_luld_top200.dbn.zst"
    status_file = DATABENTO_DIR / "dbeq_status_luld.csv"

    if not all(f.exists() for f in [bbo_file, stats_file, status_file]):
        print("  Missing prerequisite files. Run --step status, stats, bbo first.")
        return

    # Load LULD halt times (W=1 events)
    halts = pd.read_csv(status_file)
    print(f"  W=1 halt events: {len(halts)}")

    # Load LULD bands from statistics
    import databento as db
    stats_store = db.DBNStore.from_file(str(stats_file))
    stats_df = stats_store.to_df()

    # Filter for UPPER/LOWER price limits
    band_mask = stats_df["stat_type"].isin([17, 18])  # UPPER=17, LOWER=18
    bands = stats_df[band_mask].copy()
    print(f"  LULD band records: {len(bands)}")

    # Load BBO data
    bbo_store = db.DBNStore.from_file(str(bbo_file))
    bbo_df = bbo_store.to_df()
    print(f"  BBO records: {len(bbo_df)}")

    # For each symbol-day:
    #   1. Get LULD bands (upper, lower) — these update throughout the day
    #   2. Check if best_bid <= lower_band or best_ask >= upper_band
    #   3. If so, mark as "limit state entry"
    #   4. If no halt follows within 15 seconds, classify as W=0
    #   5. If halt follows, confirm as W=1

    # This is computationally intensive — we process per-symbol
    w0_events = []
    w1_times = set()
    # Build set of (symbol, halt_minute) from status data
    # ... (implementation depends on exact data format)

    print("  W=0 identification requires per-symbol BBO + band merging.")
    print("  Run identify_w0.py for the full reconstruction.")

    # Save placeholder
    out = DATABENTO_DIR / "w0_events.csv"
    pd.DataFrame(w0_events).to_csv(out, index=False)
    print(f"  Output: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download LULD data from Databento")
    parser.add_argument(
        "--step",
        choices=["cost", "status", "stats", "bbo", "ohlcv", "w0", "all"],
        default="cost",
        help="Which step to run (default: cost estimation)",
    )
    args = parser.parse_args()

    if args.step == "w0":
        identify_w0_events()
        return

    client = get_client()

    if args.step == "cost":
        estimate_costs(client)
    elif args.step == "status":
        download_status(client)
    elif args.step == "stats":
        download_statistics(client)
    elif args.step == "bbo":
        download_bbo(client)
    elif args.step == "ohlcv":
        download_ohlcv(client)
    elif args.step == "all":
        estimate_costs(client)
        print("\nProceeding with downloads...")
        download_status(client)
        download_statistics(client)
        download_bbo(client)
        download_ohlcv(client)
        identify_w0_events()


if __name__ == "__main__":
    main()
