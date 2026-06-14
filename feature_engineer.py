"""
NFP FORECASTING SYSTEM — STEP 2: FEATURE ENGINEERING
=====================================================
Loads the raw macro data from Step 1, engineers all predictive features,
and saves a clean feature matrix ready for model training.

Features generated per indicator:
  - Lag 1, 2, 3
  - Month-over-Month change
  - 3M / 6M / 12M rolling mean
  - 6M rolling standard deviation
  - 3M momentum
  - Acceleration (change-of-change)
  - 24M rolling z-score

Plus cross-series features:
  - Yield curve spread
  - Real interest rate
  - Claims trend
  - Business cycle regime

Target variable:
  - NFP monthly change (next month) = PAYEMS.diff().shift(-1)

Usage:
    python feature_engineer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from colorama import init, Fore, Style

init(autoreset=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/processed_macro_data.parquet"
OUTPUT_PATH = "data/feature_matrix.parquet"
OUTPUT_CSV  = "data/feature_matrix.csv"

# ── Which columns to engineer features for ────────────────────────────────────
# We skip USREC (it becomes a regime label) and PAYEMS (it's the target)
SKIP_COLS = ["PAYEMS", "USREC"]


def banner(text):
    print(Fore.CYAN + "\n" + "=" * 60)
    print(Fore.CYAN + f"  {text}")
    print(Fore.CYAN + "=" * 60)


def success(text): print(Fore.GREEN + f"  ✓ {text}")
def info(text):    print(Fore.WHITE + f"  → {text}")
def warn(text):    print(Fore.YELLOW + f"  ⚠ {text}")


# =============================================================================
# STEP 1 — LOAD RAW DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    banner("LOADING RAW MACRO DATA")
    df = pd.read_parquet(INPUT_PATH)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


# =============================================================================
# STEP 2 — BUILD TARGET VARIABLE
# =============================================================================

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Target = next month's NFP change (month-over-month in PAYEMS).

    Why MoM change and not level?
      - NFP releases always report the change ("economy added 180K jobs")
      - Change is stationary → better for ML models
      - This is what markets react to

    Why shift(-1)?
      - At month T we want to PREDICT the job change for month T+1
      - So we shift the known change back one period to align
        features (available at T) with target (the T+1 number)
    """
    payems_change = df["PAYEMS"].diff()          # MoM change
    target        = payems_change.shift(-1)       # next month's change
    target.name   = "TARGET_NFP_CHANGE"
    return target


# =============================================================================
# STEP 3 — ENGINEER FEATURES FOR EVERY INDICATOR
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    banner("ENGINEERING FEATURES")

    feature_cols = [c for c in df.columns if c not in SKIP_COLS]
    all_feature_dicts = []
    total = 0

    for col in feature_cols:
        s = df[col].copy()
        col_features = {}

        # ── Lags ──────────────────────────────────────────────────────────────
        for lag in [1, 2, 3]:
            col_features[f"{col}_lag{lag}"] = s.shift(lag)
            total += 1

        # ── Month-over-Month change ────────────────────────────────────────────
        mom = s.diff()
        col_features[f"{col}_mom"] = mom.shift(1)     # lag 1 to prevent leakage
        total += 1

        # ── Rolling means ─────────────────────────────────────────────────────
        for window in [3, 6, 12]:
            col_features[f"{col}_ma{window}"] = s.shift(1).rolling(window).mean()
            total += 1

        # ── Rolling standard deviation (volatility) ────────────────────────────
        col_features[f"{col}_std6"]  = s.shift(1).rolling(6).std()
        col_features[f"{col}_mom3"]  = s.shift(1) - s.shift(4)
        col_features[f"{col}_accel"] = mom.shift(1) - mom.shift(2)
        total += 3

        # ── Rolling Z-score (24-month window) ─────────────────────────────────
        rolling_mean = s.shift(1).rolling(24).mean()
        rolling_std  = s.shift(1).rolling(24).std()
        col_features[f"{col}_zscore"] = (s.shift(1) - rolling_mean) / (rolling_std + 1e-8)

        # ── Rate of change (%) ────────────────────────────────────────────────
        col_features[f"{col}_roc"] = s.shift(1).ffill().pct_change(periods=1) * 100
        total += 2

        all_feature_dicts.append(pd.DataFrame(col_features, index=df.index))

    features = pd.concat(all_feature_dicts, axis=1)
    success(f"Generated {total} features from {len(feature_cols)} indicators")
    return features


# =============================================================================
# STEP 4 — CROSS-SERIES FEATURES
# =============================================================================

def add_cross_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    banner("ADDING CROSS-SERIES FEATURES")

    # ── Yield curve: 10Y minus 2Y (already in data as T10Y2Y, but recompute) ──
    if "DGS10" in df.columns and "DGS2" in df.columns:
        features["yield_curve_10y2y"] = df["DGS10"].shift(1) - df["DGS2"].shift(1)
        success("Yield curve spread (10Y-2Y)")

    # ── Real interest rate: Fed Funds minus CPI ───────────────────────────────
    if "FEDFUNDS" in df.columns and "CPIAUCSL" in df.columns:
        cpi_yoy = df["CPIAUCSL"].shift(1).pct_change(12) * 100
        features["real_rate"] = df["FEDFUNDS"].shift(1) - cpi_yoy
        success("Real interest rate (FEDFUNDS - CPI YoY)")

    # ── Claims trend: 4-week MoM change in jobless claims ────────────────────
    if "ICSA" in df.columns:
        features["claims_trend"] = df["ICSA"].shift(1).diff(3)
        success("Claims 3M trend")

    # ── Claims ratio: continuing / initial (rising = weakening labor) ─────────
    if "ICSA" in df.columns and "CCSA" in df.columns:
        features["claims_ratio"] = df["CCSA"].shift(1) / (df["ICSA"].shift(1) + 1e-8)
        success("Claims ratio (CCSA / ICSA)")

    # ── Inflation gap: CPI minus PCE ──────────────────────────────────────────
    if "CPIAUCSL" in df.columns and "PCEPI" in df.columns:
        features["inflation_gap"] = df["CPIAUCSL"].shift(1) - df["PCEPI"].shift(1)
        success("Inflation gap (CPI - PCE)")

    # ── ADP as direct NFP predictor (when available) ──────────────────────────
    if "ADPMNUSNERSA" in df.columns:
        adp_change = df["ADPMNUSNERSA"].diff()
        features["adp_change_lag1"] = adp_change.shift(1)
        features["adp_nfp_spread"]  = df["ADPMNUSNERSA"].shift(1) - df["PAYEMS"].shift(1)
        success("ADP change & ADP-PAYEMS spread")

    # ── PAYEMS own lags and momentum (autoregressive features) ────────────────
    payems_mom = df["PAYEMS"].diff()
    for lag in [1, 2, 3, 6]:
        features[f"payems_change_lag{lag}"] = payems_mom.shift(lag)
    features["payems_3m_avg_change"] = payems_mom.shift(1).rolling(3).mean()
    features["payems_6m_avg_change"] = payems_mom.shift(1).rolling(6).mean()
    success("PAYEMS autoregressive features (own lags)")

    return features


# =============================================================================
# STEP 5 — CALENDAR & REGIME FEATURES
# =============================================================================

def add_calendar_regime_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    banner("ADDING CALENDAR & REGIME FEATURES")

    idx = df.index

    # ── Month of year (seasonality) ───────────────────────────────────────────
    features["month"]   = idx.month
    features["quarter"] = idx.quarter

    # ── Month sine/cosine encoding (captures cyclical seasonality better) ─────
    features["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    success("Calendar features: month, quarter, sin/cos encoding")

    # ── Recession regime ──────────────────────────────────────────────────────
    if "USREC" in df.columns:
        features["is_recession"]         = df["USREC"].shift(1).fillna(0)
        features["months_since_recession"] = (
            df["USREC"].shift(1)
            .fillna(0)
            .groupby((df["USREC"].shift(1) != df["USREC"].shift(1).shift()).cumsum())
            .cumcount()
        )
        success("Recession indicator & months since last recession")

    # ── Rate hike / cut cycle ─────────────────────────────────────────────────
    if "FEDFUNDS" in df.columns:
        ff_change = df["FEDFUNDS"].diff().shift(1)
        features["rate_hiking"]  = (ff_change > 0).astype(int)
        features["rate_cutting"] = (ff_change < 0).astype(int)
        features["rate_change"]  = ff_change
        success("Rate cycle: hiking / cutting / neutral")

    # ── VIX regime: high stress vs normal ─────────────────────────────────────
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"].shift(1)
        features["vix_high_stress"] = (vix > 25).astype(int)
        features["vix_extreme"]     = (vix > 40).astype(int)
        success("VIX stress regime flags")

    return features


# =============================================================================
# STEP 6 — ASSEMBLE FINAL MATRIX
# =============================================================================

def assemble_final_matrix(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    banner("ASSEMBLING FINAL FEATURE MATRIX")

    df_final = features.copy()
    df_final["TARGET_NFP_CHANGE"] = target

    # Drop rows where target is missing (future months + first few rows)
    rows_before = len(df_final)
    df_final.dropna(subset=["TARGET_NFP_CHANGE"], inplace=True)
    rows_after = len(df_final)
    info(f"Dropped {rows_before - rows_after} rows with missing target")

    # Replace infinite values with NaN to avoid errors in models
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Report missing values in features
    missing = df_final.drop(columns=["TARGET_NFP_CHANGE"]).isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        warn(f"{len(missing)} feature columns still have NaN (from rolling window warmup or inf replacement)")
        info("These will be handled by XGBoost natively — no action needed")

    success(f"Final matrix: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    info(f"  Feature columns : {df_final.shape[1] - 1}")
    info(f"  Target rows     : {df_final['TARGET_NFP_CHANGE'].notna().sum()}")
    info(f"  Date range      : {df_final.index.min().date()} → {df_final.index.max().date()}")

    return df_final


# =============================================================================
# STEP 7 — SAVE & REPORT
# =============================================================================

def save_and_report(df_final: pd.DataFrame):
    banner("SAVING FEATURE MATRIX")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH)
    df_final.to_csv(OUTPUT_CSV)

    success(f"Parquet saved → {OUTPUT_PATH}")
    success(f"CSV saved     → {OUTPUT_CSV}")

    # ── Summary of feature groups ─────────────────────────────────────────────
    banner("FEATURE ENGINEERING COMPLETE — SUMMARY")

    cols = [c for c in df_final.columns if c != "TARGET_NFP_CHANGE"]

    lag_cols      = [c for c in cols if "_lag"    in c]
    mom_cols      = [c for c in cols if "_mom"    in c and "3" not in c]
    ma_cols       = [c for c in cols if "_ma"     in c]
    std_cols      = [c for c in cols if "_std"    in c]
    zscore_cols   = [c for c in cols if "_zscore" in c]
    accel_cols    = [c for c in cols if "_accel"  in c]
    roc_cols      = [c for c in cols if "_roc"    in c]
    cross_cols    = [c for c in cols if c in [
        "yield_curve_10y2y","real_rate","claims_trend","claims_ratio",
        "inflation_gap","adp_change_lag1","adp_nfp_spread"
    ] + [f"payems_change_lag{i}" for i in [1,2,3,6]]
      + ["payems_3m_avg_change","payems_6m_avg_change"]]
    cal_cols      = [c for c in cols if c in [
        "month","quarter","month_sin","month_cos",
        "is_recession","months_since_recession",
        "rate_hiking","rate_cutting","rate_change",
        "vix_high_stress","vix_extreme"
    ]]

    print(f"\n  {'Feature Type':<35} {'Count':>6}")
    print(f"  {'-'*45}")
    print(f"  {'Lag features (1,2,3)':<35} {len(lag_cols):>6}")
    print(f"  {'MoM change':<35} {len(mom_cols):>6}")
    print(f"  {'Rolling means (3M,6M,12M)':<35} {len(ma_cols):>6}")
    print(f"  {'Rolling std dev (6M)':<35} {len(std_cols):>6}")
    print(f"  {'Z-scores (24M)':<35} {len(zscore_cols):>6}")
    print(f"  {'Acceleration':<35} {len(accel_cols):>6}")
    print(f"  {'Rate of change (%)':<35} {len(roc_cols):>6}")
    print(f"  {'Cross-series features':<35} {len(cross_cols):>6}")
    print(f"  {'Calendar & regime features':<35} {len(cal_cols):>6}")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL FEATURES':<35} {len(cols):>6}")
    print(f"  {'TARGET rows':<35} {df_final['TARGET_NFP_CHANGE'].notna().sum():>6}")
    print()

    # Target distribution
    t = df_final["TARGET_NFP_CHANGE"].dropna()
    print(Fore.CYAN + "  TARGET — NFP Monthly Change Distribution:")
    print(f"  {'  Mean':<35} {t.mean():>+8.1f}K")
    print(f"  {'  Std Dev':<35} {t.std():>8.1f}K")
    print(f"  {'  Min (worst month)':<35} {t.min():>+8.1f}K")
    print(f"  {'  Max (best month)':<35} {t.max():>+8.1f}K")
    print(f"  {'  Median':<35} {t.median():>+8.1f}K")

    banner("STEP 2 COMPLETE — READY FOR MODEL TRAINING")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("NFP FORECASTING SYSTEM — FEATURE ENGINEER")

    # Load
    df = load_data()

    # Build target
    target = build_target(df)

    # Engineer features
    features = engineer_features(df)
    features = add_cross_features(df, features)
    features = add_calendar_regime_features(df, features)

    # Assemble
    df_final = assemble_final_matrix(features, target)

    # Save
    save_and_report(df_final)


if __name__ == "__main__":
    main()
