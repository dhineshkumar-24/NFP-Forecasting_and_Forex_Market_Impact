import time
import logging
import datetime
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from fredapi import Fred
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Load config parameters
try:
    from config import (
        FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE,
        RAW_DATA_DIR, PROCESSED_DATA_PATH, WEEKLY_SERIES, DAILY_SERIES, logger
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from config import (
        FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE,
        RAW_DATA_DIR, PROCESSED_DATA_PATH, WEEKLY_SERIES, DAILY_SERIES, logger
    )

# Initialize colorama
colorama.init(autoreset=True)

def generate_raw_mock_series(series_id: str, start_date: str) -> pd.Series:
    """
    Generates a raw mock pandas Series for a series_id at the appropriate raw frequency.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(datetime.date.today())
    
    if series_id in DAILY_SERIES:
        freq = "D"
    elif series_id in WEEKLY_SERIES:
        freq = "W-SAT"
    else:
        freq = "MS"
        
    dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    n_periods = len(dates)
    if n_periods == 0:
        return pd.Series(dtype=float)
        
    np.random.seed(hash(series_id) % (2**32 - 1))
    
    # Mock realistic trends and recessions
    if series_id == "PAYEMS":
        base = np.linspace(131000.0, 162000.0, n_periods)
        for idx, dt in enumerate(dates):
            if dt.year in [2008, 2009]:
                base[idx] -= 6000
            elif dt.year == 2020 and dt.month >= 3 and dt.month <= 6:
                base[idx] -= 15000
        values = base + np.random.normal(0, 100, n_periods)
    elif series_id == "UNRATE":
        base = np.full(n_periods, 5.0)
        for idx, dt in enumerate(dates):
            if dt.year in [2008, 2009]:
                base[idx] = 8.5
            elif dt.year == 2020 and dt.month >= 3 and dt.month <= 6:
                base[idx] = 13.0
        values = np.clip(base + np.random.normal(0, 0.2, n_periods), 3.0, 15.0)
    elif series_id == "ICSA":
        base = np.full(n_periods, 300.0)
        for idx, dt in enumerate(dates):
            if dt.year in [2008, 2009]:
                base[idx] = 550.0
            elif dt.year == 2020 and dt.month >= 3 and dt.month <= 6:
                base[idx] = 4000.0
        values = np.clip(base + np.random.normal(0, 20, n_periods), 150.0, 7000.0)
    else:
        values = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, n_periods))
        
    return pd.Series(values, index=dates)

def main():
    parser = argparse.ArgumentParser(description="FRED Data Collector & Preprocessor")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode generating synthetic data")
    args = parser.parse_args()

    # Create directories if they do not exist
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_parquet_path = Path(PROCESSED_DATA_PATH)
    processed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    processed_csv_path = processed_parquet_path.with_suffix(".csv")

    use_mock = args.mock
    fred = None

    if not use_mock:
        if not FRED_API_KEY or FRED_API_KEY == "your_actual_api_key_here":
            logger.warning("Valid FRED API key is not configured. Falling back to Mock mode.")
            use_mock = True
        else:
            try:
                fred = Fred(api_key=FRED_API_KEY)
                logger.info("Connected to FRED API successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to FRED API: {e}. Falling back to Mock mode.")
                use_mock = True

    downloaded_series = {}

    print(Fore.CYAN + f"\nStarting data pipeline ({'Mock Mode' if use_mock else 'FRED API Mode'})...")

    # Loop with tqdm progress bar
    progress_bar = tqdm(FRED_SERIES.keys(), desc="Processing indicators", unit="series")
    
    for series_id in progress_bar:
        description = FRED_SERIES[series_id][1]
        progress_bar.set_postfix_str(f"ID: {series_id}")

        try:
            # 4. Fetch or mock the series
            if use_mock:
                series = generate_mock_series = generate_raw_mock_series(series_id, START_DATE)
            else:
                series = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
                # Respect rate limit
                time.sleep(0.12)
            
            if series is None or series.empty:
                logger.warning(f"No data retrieved for {series_id}.")
                continue

            # Ensure index is datetime
            series.index = pd.to_datetime(series.index)
            
            # 5. Resample to month-end frequency
            if series_id in DAILY_SERIES:
                series.index = pd.to_datetime(series.index)
                resampled = series.resample("ME").last()
                resampled = resampled.dropna()
            elif series_id in WEEKLY_SERIES:
                resampled = series.resample("ME").mean()
            else:
                resampled = series.copy()
                resampled.index = resampled.index.to_period("M").to_timestamp("M")
                resampled = resampled.groupby(level=0).last()

            # 6. Save individual series as parquet
            df_series = resampled.to_frame(name=series_id)
            df_series.to_parquet(raw_dir / f"{series_id}.parquet")

            downloaded_series[series_id] = resampled

        except Exception as e:
            logger.error(f"Error processing series {series_id} ({description}): {e}")
            print(Fore.RED + f"Error processing {series_id}: {e}")
            continue

    if not downloaded_series:
        logger.error("No data was collected or processed. Pipeline aborting.")
        print(Fore.RED + "Pipeline failed: no series processed.")
        return

    try:
        # 7. Merge all series into one wide DataFrame
        print(Fore.CYAN + "\nMerging and consolidating indicators...")
        resampled_dfs = [s.to_frame(name=series_id) for series_id, s in downloaded_series.items()]
        wide_df = pd.concat(resampled_dfs, axis=1)
        wide_df.index.name = "date"

        # 8. Forward-fill missing values up to 3 periods (applies to all columns, including DJIA and BAMLH0A0HYM2)
        wide_df = wide_df.ffill(limit=3)

        # 9. Drops rows where PAYEMS is missing
        if "PAYEMS" in wide_df.columns:
            wide_df = wide_df.dropna(subset=["PAYEMS"])
        else:
            logger.warning("PAYEMS (target) not found in dataset. Rows with missing PAYEMS cannot be dropped.")

        # Reset index to make 'date' a column
        final_df = wide_df.reset_index()

        # 10. Save processed DataFrame as parquet (keep date as index)
        wide_df.to_parquet(processed_parquet_path)
        
        # 11. Save CSV copy
        final_df.to_csv(processed_csv_path, index=False)

        print(Fore.GREEN + Style.BRIGHT + "\n=== PIPELINE SUCCESSFUL ===")
        print(f"Parquet saved to: {processed_parquet_path}")
        print(f"CSV saved to:     {processed_csv_path}")

        # 12. Prints validation report
        print(Fore.GREEN + Style.BRIGHT + "\n=== VALIDATION REPORT ===")
        print(f"Total Rows: {len(final_df)}")
        print(f"Total Columns: {len(final_df.columns)}")
        
        if not final_df.empty:
            start_date_str = final_df["date"].min().strftime("%Y-%m-%d")
            end_date_str = final_df["date"].max().strftime("%Y-%m-%d")
            print(f"Date Range: {start_date_str} to {end_date_str}")
        else:
            print(Fore.YELLOW + "Date Range: Empty dataset")

        # Missing values per indicator
        print("\nMissing Values Count:")
        missing_counts = wide_df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(Fore.YELLOW + f"  {col}: {count} missing")
            else:
                print(Fore.GREEN + f"  {col}: 0 missing")

        # PAYEMS statistics
        if "PAYEMS" in final_df.columns:
            payems_data = final_df["PAYEMS"]
            print("\nPAYEMS Target Statistics:")
            print(f"  Mean:  {payems_data.mean():,.2f}")
            print(f"  Min:   {payems_data.min():,.2f}")
            print(f"  Max:   {payems_data.max():,.2f}")
            print(f"  Last:  {payems_data.iloc[-1]:,.2f}")

    except Exception as e:
        logger.error(f"Error merging or saving final datasets: {e}")
        print(Fore.RED + f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()


