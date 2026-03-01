# CL-F1 Empirical Data: LULD Pause Events

## Data Sources

### 1. NYSE Trading Halts (DOWNLOADED)

**Source:** NYSE API — `https://www.nyse.com/api/trade-halts/historical/download`
**Access:** Free, no authentication, direct CSV download
**Coverage:** 2020-01-01 to present
**Location:** `raw/nyse_halts/`

| File | Rows | LULD Events |
|------|------|-------------|
| `nyse_halts_2020.csv` | 14,929 | 14,076 |
| `nyse_halts_2021.csv` | 5,634 | 4,682 |
| `nyse_halts_2022.csv` | 6,893 | 5,751 |
| `nyse_halts_2023.csv` | 9,385 | 7,772 |
| `nyse_halts_2024.csv` | 11,219 | 8,882 |
| `nyse_halts_2025.csv` | 13,080 | 10,749 |
| **`luld_events_2020_2025.csv`** | — | **51,912** |

**Fields:** `Halt Date, Halt Time, Symbol, Name, Exchange, Reason, Resume Date, NYSE Resume Time`

**Key statistics (2024):**
- 8,882 LULD events across 1,515 unique symbols
- 85% Nasdaq-listed, 7% NYSE American, 7% NYSE
- Median halt duration: 300s (5 min, standard LULD)
- Mean halt duration: 427s (tail from extended halts)
- Monthly: 500–1,300 events/month

### 2. Databento Tick-Level Data (DOWNLOADED)

**Source:** Databento API — datasets `DBEQ.BASIC` and `XNAS.ITCH`
**Access:** $125 free credit; actual spend ~$3 for targeted downloads
**Coverage:** 2024-01-01 to 2024-12-31
**Location:** `raw/databento/`

| File | Dataset | Schema | Symbols | Size | Records | Description |
|------|---------|--------|---------|------|---------|-------------|
| `dbeq_status_top50.dbn.zst` | DBEQ.BASIC | status | top 50 | 2.0 MB | 250,289 | Halt/resume events (microsecond timestamps) |
| `dbeq_status_top50.csv` | — | — | — | 25 MB | — | CSV of above |
| `xnas_status_top50.dbn.zst` | XNAS.ITCH | status | top 50 | 0.4 MB | 37,462 | Nasdaq-specific halt details |
| `xnas_statistics_top50.dbn.zst` | XNAS.ITCH | statistics | top 50 | 0.4 MB | 17,577 | Opening/closing/uncrossing prices |
| `dbeq_ohlcv1m_top50.dbn.zst` | DBEQ.BASIC | ohlcv-1m | top 50 | 4.6 MB | ~200K | 1-minute OHLCV for outcome construction |
| `dbeq_mbp1_top20.dbn.zst` | DBEQ.BASIC | mbp-1 | top 20 | 258 MB | ~50M+ | **Top-of-book quotes (bid/ask) for W=0 detection** |
| `dbeq_trades_top20.dbn.zst` | DBEQ.BASIC | trades | top 20 | 9.1 MB | ~1M+ | Trades for reference price calculation |
| `dbeq_status_all.dbn.zst` | DBEQ.BASIC | status | ALL | 1.1 GB | ~25M | All instruments, all status messages (2023-04 to 2025) |

**Top 20 LULD symbols (by 2024 halt frequency):**
RGC, SWIN, PCTTU, QMMM, LPA, INHD, TOP, AFJK, FLYE, NVNI,
HOLO, SKK, DRCT, GME, JBDI, XCH, FMTO, ASPC, ENGS, PGHL

**LULD events in Databento status data:**
- DBEQ.BASIC: 4,027 LULD events (reason=50, action=8 HALT), 37 unique symbols
- XNAS.ITCH: 1,155 LULD events (reason=50, action=9 PAUSE), Nasdaq-listed only
- Note: DBEQ.BASIC shows ~3 records per halt (one per venue: IEX, NYSE Chicago, NYSE National)

### 3. Covariates (TO DOWNLOAD)

| Variable | Source | Granularity | Cost |
|----------|--------|-------------|------|
| VIX | FRED / CBOE | Daily | Free |
| Market cap, sector | Alpha Vantage / FMP | Daily | Free |
| Volume, spread | Computed from Databento | Tick-level | Already have |
| Short interest | FINRA | Bi-monthly | Free |

## Treatment Assignment: W=0 vs W=1

### W=1 Identification (Full Halt — DONE)

From status data: `action=8 (HALT)` or `action=9 (PAUSE)` with `reason=50 (LULD_PAUSE)`.
Confirmed against NYSE halt file. Microsecond timestamps from Databento.

### W=0 Identification (Limit State Resolved — PIPELINE READY)

W=0 events (limit state entries that resolved within 15 seconds without triggering a halt) are **not** recorded as status messages. They must be reconstructed from quote data:

**Pipeline:**
1. **Reference price**: From trades data, compute 5-minute arithmetic mean of trade prices
2. **LULD bands**: Apply tier-specific percentages to reference price
   - Tier 1 (S&P 500, Russell 1000, some ETFs): ±5% (±10% during first/last 15 min)
   - Tier 2 (other NMS, price ≥ $3): ±10% (±20% during first/last 15 min)
   - Tier 2 (price $0.75–$3): ±20% (±40% during first/last 15 min)
   - Tier 2 (price < $0.75): lesser of ±75% or $0.15
3. **Limit state entry**: Identify timestamps where `ask_px ≥ upper_band` or `bid_px ≤ lower_band` in MBP-1 data
4. **Cross-reference**: Check whether a HALT status message follows within 15 seconds
   - If halt follows → W=1 (confirmation, already captured)
   - If no halt → **W=0** (limit state resolved)
5. **Output**: CSV with `(symbol, ts_limit_state, W, duration_seconds, pre_price, post_price)`

**Data sufficiency:**
- MBP-1 (258 MB, top 20 symbols, 2024): bid/ask at every quote update
- Trades (9.1 MB, top 20 symbols, 2024): for reference price calculation
- Status (2 MB, top 50 symbols, 2024): W=1 halt timestamps

## Empirical Design Options

### Design A: Full halt vs matched non-halt (FREE DATA ONLY)

- **W=1**: Stock i experienced LULD halt at time t
- **W=0**: Matched control stock (same sector, similar pre-halt volatility) with NO halt on date t
- **Outcome Y**: Post-halt 5/15/30-min return, realized volatility, or price impact
- **Covariates X**: Pre-halt 1-min returns, volume, VIX, market cap, sector
- **Matching**: Propensity score from pre-halt characteristics

### Design B: Full halt vs limit-state-resolved (TICK DATA — HAVE IT)

The paper's primary design. Uses Databento MBP-1 + trades + status data.
- **W=1**: HALT status with LULD_PAUSE reason (from status schema)
- **W=0**: Limit state entry (NBBO touches LULD band) that resolves within 15 seconds
- **Outcome Y**: Post-event price trajectory, realized vol, spread recovery
- **Covariates X**: Pre-event 5-min returns, volume, spread, distance to band
- **Identification**: Conditional on limit state entry, assignment to halt vs resolved is quasi-random (depends on 15-second order flow)

### Design C: RD around LULD band boundary

- Running variable: distance of price from LULD band boundary
- Treatment: crossing the band triggers limit state
- Sharp RD at the band boundary
- Requires: trade/quote data + LULD band calculations (have both)

## Download Commands

### Refresh NYSE halt data
```bash
for year in 2020 2021 2022 2023 2024 2025; do
    curl -sL "https://www.nyse.com/api/trade-halts/historical/download?haltDateFrom=${year}-01-01&haltDateTo=${year}-12-31&symbol=" \
        -o raw/nyse_halts/nyse_halts_${year}.csv
done
```

### Databento (requires API key)
```bash
export DATABENTO_API_KEY=db-XXXXXXXX
python download_luld_databento.py --step cost   # estimate before downloading
python download_luld_databento.py --step all    # download all schemas
```

### Databento credit usage
- Initial $125 credit
- Targeted downloads (top 20-50 symbols, 2024): ~$3
- Aborted ALL_SYMBOLS status: ~$78 (partial, file saved)
- Remaining: ~$44
