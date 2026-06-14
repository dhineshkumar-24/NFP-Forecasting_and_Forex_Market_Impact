"""
NFP FORECASTING SYSTEM — STAGE 3: MARKET REACTION MODEL
========================================================
Uses conditional historical statistics to show how 5 key
markets have historically moved on NFP release day, grouped
by surprise magnitude buckets.

Surprise Buckets:
  Bucket 1: surprise < -50      (LARGE MISS)
  Bucket 2: -50 <= surprise < 0 (SMALL MISS)
  Bucket 3: 0 <= surprise < 50  (SMALL BEAT)
  Bucket 4: surprise >= 50      (LARGE BEAT)

Assets covered:
  Gold (GC=F), EUR/USD (EURUSD=X), DXY (DX-Y.NYB),
  10Y Treasury (^TNX), S&P 500 (^GSPC)

Usage:
    python market_reaction_model.py

Run AFTER:
    python surprise_model.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from colorama import init, Fore, Style

import yfinance as yf

warnings.filterwarnings("ignore")
init(autoreset=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
CONSENSUS_PATH  = "data/consensus_data.csv"
MR_RESULTS_PATH = "data/market_reaction_results.csv"

# ── Assets ────────────────────────────────────────────────────────────────────
ASSETS = {
    "Gold"     : "GC=F",
    "EURUSD"   : "EURUSD=X",
    "DXY"      : "DX-Y.NYB",
    "10Y Yield": "^TNX",
    "SP500"    : "^GSPC",
}

# ── Surprise buckets ──────────────────────────────────────────────────────────
BUCKETS = [
    ("LARGE MISS  (< -50K)",  lambda s: s <  -50),
    ("SMALL MISS  (-50K–0K)", lambda s: -50 <= s < 0),
    ("SMALL BEAT  (0K–+50K)", lambda s: 0  <= s < 50),
    ("LARGE BEAT  (> +50K)",  lambda s: s >= 50),
]

COVID_MONTHS = [
    "2020-03-31", "2020-04-30", "2020-05-31",
    "2020-06-30", "2020-07-31", "2020-08-31",
]


def banner(text, color=Fore.CYAN):
    print(color + "\n" + "=" * 60)
    print(color + f"  {text}")
    print(color + "=" * 60)

def success(t): print(Fore.GREEN  + f"  \u2713 {t}")
def info(t):    print(Fore.WHITE  + f"  -> {t}")
def warn(t):    print(Fore.YELLOW + f"  ! {t}")


# =============================================================================
# STEP 1 — BUILD NFP RELEASE DATE CALENDAR
# =============================================================================

def get_nfp_release_date(reference_month: pd.Timestamp) -> pd.Timestamp:
    """
    NFP for month M is released on the first Friday of month M+1.
    reference_month is the month-end of the jobs month (e.g. 2026-05-31).
    """
    if reference_month.month == 12:
        first_of_next = pd.Timestamp(reference_month.year + 1, 1, 1)
    else:
        first_of_next = pd.Timestamp(reference_month.year, reference_month.month + 1, 1)

    day_of_week = first_of_next.dayofweek   # Monday=0, Friday=4
    days_until_friday = (4 - day_of_week) % 7
    return first_of_next + timedelta(days=days_until_friday)


# =============================================================================
# STEP 2 — DOWNLOAD MARKET DATA ON NFP DAYS
# =============================================================================

def download_nfp_day_returns(nfp_release_dates: list) -> pd.DataFrame:
    banner("DOWNLOADING MARKET DATA (NFP DAYS)")

    records = []

    for asset_name, ticker in ASSETS.items():
        info(f"Downloading {asset_name} ({ticker})...")
        try:
            raw = yf.download(
                ticker,
                start="2014-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                warn(f"{asset_name}: No data returned")
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.index = pd.to_datetime(raw.index).tz_localize(None)

            for jobs_month_end, release_dt in nfp_release_dates:
                # Find the trading day on or after the NFP release date
                window = raw.loc[
                    (raw.index >= release_dt) &
                    (raw.index <= release_dt + timedelta(days=3))
                ]
                if window.empty:
                    continue

                release_day = window.index[0]

                # Previous close (day before NFP release)
                prev_window = raw.loc[raw.index < release_day]
                if prev_window.empty:
                    continue

                prev_close = float(prev_window["Close"].iloc[-1])
                nfp_close  = float(window["Close"].iloc[0])

                daily_return = (nfp_close - prev_close) / prev_close * 100

                records.append({
                    "jobs_month"  : jobs_month_end,
                    "release_date": release_day,
                    "asset"       : asset_name,
                    "return_pct"  : round(daily_return, 4),
                })

        except Exception as e:
            warn(f"{asset_name}: Download failed — {e}")

    if not records:
        warn("No market data collected.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Pivot to wide: one row per jobs_month, one column per asset
    wide = df.pivot_table(
        index="jobs_month",
        columns="asset",
        values="return_pct",
        aggfunc="first",
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    success(f"Collected {len(wide)} NFP day observations across {len(ASSETS)} assets")
    return wide


# =============================================================================
# STEP 3 — BUILD MERGED DATASET
# =============================================================================

def build_dataset(market_returns: pd.DataFrame) -> pd.DataFrame:
    banner("MERGING CONSENSUS & MARKET DATA")

    consensus = pd.read_csv(CONSENSUS_PATH, parse_dates=["date"])
    consensus = consensus.set_index("date")
    consensus.index = pd.to_datetime(consensus.index)

    # Remove COVID distortion months
    covid_idx = pd.to_datetime(COVID_MONTHS)
    consensus = consensus[~consensus.index.isin(covid_idx)]

    df = consensus.join(market_returns, how="inner")
    df = df.dropna(subset=list(ASSETS.keys()), how="all")

    info(f"Merged dataset: {len(df)} months")
    info(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    return df


# =============================================================================
# STEP 4 — CALCULATE BUCKET STATISTICS
# =============================================================================

def calc_bucket_stats(df: pd.DataFrame) -> dict:
    """
    For each surprise bucket and each asset, calculate:
      up_rate, avg_move, median_move, best_case (p75), worst_case (p25), count
    """
    bucket_stats = {}

    for bucket_label, bucket_fn in BUCKETS:
        mask = df["surprise"].apply(bucket_fn)
        subset = df[mask]
        n = len(subset)

        asset_stats = {}
        for asset in ASSETS.keys():
            if asset not in subset.columns:
                continue
            col = subset[asset].dropna()
            if col.empty:
                continue
            asset_stats[asset] = {
                "up_rate"     : round(float((col > 0).mean() * 100), 1),
                "avg_move"    : round(float(col.mean()), 3),
                "median_move" : round(float(col.median()), 3),
                "best_case"   : round(float(col.quantile(0.75)), 3),
                "worst_case"  : round(float(col.quantile(0.25)), 3),
                "sample_count": int(col.count()),
            }

        bucket_stats[bucket_label] = {
            "n"          : n,
            "asset_stats": asset_stats,
        }

    return bucket_stats


# =============================================================================
# STEP 5 — PRINT FULL STATISTICS TABLE
# =============================================================================

def print_bucket_table(bucket_stats: dict):
    banner("HISTORICAL MARKET REACTIONS BY SURPRISE BUCKET")

    c = Fore.CYAN
    w = Fore.WHITE
    g = Fore.GREEN
    r = Fore.RED
    y = Fore.YELLOW

    for bucket_label, data in bucket_stats.items():
        n = data["n"]
        stats = data["asset_stats"]

        print()
        print(y + f"  Bucket: {bucket_label}   |   {n} observations")
        print(c + f"  {'Asset':<12} {'Up%':>6} {'Avg Move':>10} {'Median':>10} "
                  f"{'25th pct':>10} {'75th pct':>10} {'N':>5}")
        print(c + "  " + "-" * 65)

        if not stats:
            print(w + "  (no data)")
            continue

        for asset, s in stats.items():
            up_col  = g if s["up_rate"] >= 55 else (r if s["up_rate"] <= 45 else w)
            avg_col = g if s["avg_move"] > 0 else (r if s["avg_move"] < 0 else w)
            print(
                w  + f"  {asset:<12}"
                + up_col  + f" {s['up_rate']:>5.0f}%"
                + avg_col + f" {s['avg_move']:>+9.2f}%"
                + w       + f" {s['median_move']:>+9.2f}%"
                          + f" {s['worst_case']:>+9.2f}%"
                          + f" {s['best_case']:>+9.2f}%"
                          + f" {s['sample_count']:>5}"
            )


# =============================================================================
# STEP 6 — LIVE FORECAST BOX
# =============================================================================

def live_forecast_box(
    bucket_stats: dict,
    predicted_surprise: float,
    beat_prob: float,
):
    banner("LIVE NFP DAY HISTORICAL PATTERN FORECAST", Fore.GREEN)

    # Identify which bucket the predicted surprise falls into
    matched_label = None
    matched_data  = None
    for bucket_label, bucket_fn in BUCKETS:
        if bucket_fn(predicted_surprise):
            matched_label = bucket_label.strip()
            matched_data  = bucket_stats[bucket_label]
            break

    if matched_data is None:
        warn("Could not match prediction to a bucket.")
        return

    stats = matched_data["asset_stats"]
    n     = matched_data["n"]

    # NFP release date (first Friday of next month from today)
    today    = pd.Timestamp.today()
    nfp_date = get_nfp_release_date(
        pd.Timestamp(today.year, today.month, 1) - pd.offsets.MonthEnd(1)
    )

    c = Fore.CYAN
    w = Fore.WHITE
    g = Fore.GREEN
    r = Fore.RED
    y = Fore.YELLOW

    print()
    print(c + "=" * 62)
    print(c + f"  NFP DAY HISTORICAL PATTERN -- {nfp_date.strftime('%B %d, %Y')}")
    print(c + "=" * 62)
    print(w + f"  Surprise bucket  : {matched_label}")
    print(w + f"  Predicted surprise: {predicted_surprise:>+.0f}K jobs")
    print(w + f"  Beat probability  : {beat_prob:.0%}")
    print(w + f"  Based on {n} historical observations")
    print(c + "  " + "-" * 58)
    print(w + f"  {'Asset':<12} {'Up%':>5}  {'Avg Move':>10}  {'Range (25th-75th)':>22}")
    print(c + "  " + "-" * 58)

    for asset in ASSETS.keys():
        if asset not in stats:
            continue
        s       = stats[asset]
        up_col  = g if s["up_rate"] >= 55 else (r if s["up_rate"] <= 45 else w)
        avg_col = g if s["avg_move"] > 0 else (r if s["avg_move"] < 0 else w)
        rng     = f"{s['worst_case']:>+.2f}% to {s['best_case']:>+.2f}%"
        print(
            w       + f"  {asset:<12}"
            + up_col  + f" {s['up_rate']:>4.0f}%"
            + avg_col + f"  {s['avg_move']:>+9.2f}%"
            + w       + f"  {rng:>24}"
        )

    print(c + "=" * 62)
    print()
    print(y + "  DISCLAIMER:")
    print(w + "  These are historical averages only. Past patterns do not")
    print(w + "  guarantee future reactions. Use for context, not as")
    print(w + "  trading signals.")
    print()


# =============================================================================
# STEP 7 — SAVE RESULTS
# =============================================================================

def save_results(df: pd.DataFrame, bucket_stats: dict):
    rows = []
    for bucket_label, data in bucket_stats.items():
        for asset, s in data["asset_stats"].items():
            rows.append({
                "bucket"       : bucket_label.strip(),
                "asset"        : asset,
                "n_obs"        : data["n"],
                **s,
            })
    out = pd.DataFrame(rows)
    out.to_csv(MR_RESULTS_PATH, index=False)
    success(f"Bucket statistics saved to {MR_RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("NFP FORECASTING SYSTEM -- STAGE 3: MARKET REACTION")
    info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Build (jobs_month_end, release_date) pairs
    consensus = pd.read_csv(CONSENSUS_PATH, parse_dates=["date"])
    nfp_pairs = [
        (pd.Timestamp(row["date"]), get_nfp_release_date(pd.Timestamp(row["date"])))
        for _, row in consensus.iterrows()
    ]
    info(f"Built {len(nfp_pairs)} NFP release date pairs")

    # Download NFP-day market returns
    market_returns = download_nfp_day_returns(nfp_pairs)
    if market_returns.empty:
        warn("No market data returned — check internet connection.")
        return

    # Merge with consensus
    df = build_dataset(market_returns)
    if df.empty:
        warn("Merged dataset is empty — cannot continue.")
        return

    # Calculate bucket statistics
    bucket_stats = calc_bucket_stats(df)

    # Print full statistics table
    print_bucket_table(bucket_stats)

    # Save results CSV
    save_results(df, bucket_stats)

    # ── Live forecast ──────────────────────────────────────────────────────
    banner("ENTER STAGE 2 OUTPUTS FOR LIVE FORECAST")
    print(Fore.YELLOW + """
  Enter the predicted surprise and beat probability from
  surprise_model.py to see the historical pattern for that
  surprise bucket.
    """)

    try:
        surp_raw  = input(Fore.CYAN + "  Predicted surprise in K (e.g. 85 or -30): ").strip()
        prob_raw  = input(Fore.CYAN + "  Beat probability 0-1 (e.g. 0.94): ").strip()

        predicted_surprise = float(surp_raw)
        beat_prob          = float(prob_raw)

        live_forecast_box(bucket_stats, predicted_surprise, beat_prob)

    except (ValueError, KeyboardInterrupt):
        warn("Skipping live forecast.")

    banner("STAGE 3 COMPLETE", Fore.GREEN)
    print(Fore.GREEN + """
  Monthly workflow (Thursday before NFP Friday):
  1. python data_collector.py
  2. python feature_engineer.py
  3. python nowcaster.py
  4. python surprise_model.py
  5. python market_reaction_model.py
    """)


if __name__ == "__main__":
    main()