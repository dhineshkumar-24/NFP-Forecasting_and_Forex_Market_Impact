"""
NFP FORECASTING SYSTEM — STEP 4: NOWCASTER
===========================================
Generates a live NFP forecast for the upcoming release.
Run this on the Thursday before NFP Friday, after ADP is released.

What it does:
  1. Loads the trained model (models/nfp_model.pkl)
  2. Loads the latest feature matrix
  3. Takes the most recent row (current month's features)
  4. Applies bias correction from walk-forward history
  5. Generates prediction + confidence intervals
  6. Prints a clean forecast output
  7. Saves forecast to forecast_history.csv

Usage:
    python nowcaster.py

Run this AFTER:
    python data_collector.py
    python feature_engineer.py
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style

warnings.filterwarnings("ignore")
init(autoreset=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH = "data/feature_matrix.parquet"
WF_RESULTS_PATH     = "data/walk_forward_results.csv"
MODEL_PATH          = "models/nfp_model.pkl"
IMPUTER_PATH        = "models/imputer.pkl"
METADATA_PATH       = "models/model_metadata.json"
HISTORY_PATH        = "data/forecast_history.csv"

# ── COVID months to exclude from bias calculation ─────────────────────────────
COVID_MONTHS = [
    "2020-03-31", "2020-04-30", "2020-05-31",
    "2020-06-30", "2020-07-31", "2020-08-31",
]


def banner(text, color=Fore.CYAN):
    print(color + "\n" + "=" * 60)
    print(color + f"  {text}")
    print(color + "=" * 60)


def success(text): print(Fore.GREEN  + f"  ✓ {text}")
def info(text):    print(Fore.WHITE  + f"  → {text}")
def warn(text):    print(Fore.YELLOW + f"  ⚠ {text}")


# =============================================================================
# STEP 1 — LOAD MODEL & METADATA
# =============================================================================

def load_model():
    banner("LOADING MODEL")

    with open(MODEL_PATH,   "rb") as f: model   = pickle.load(f)
    with open(IMPUTER_PATH, "rb") as f: imputer = pickle.load(f)
    with open(METADATA_PATH, "r") as f: meta    = json.load(f)

    success(f"Model loaded  — trained on {meta['n_months']} months")
    success(f"Features      — {meta['n_features']} columns")
    info(f"Training range: {meta['train_start']} → {meta['train_end']}")

    return model, imputer, meta


# =============================================================================
# STEP 2 — LOAD FEATURES & BUILD PREDICTION ROW
# =============================================================================

def load_latest_features():
    banner("LOADING LATEST FEATURES")

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Add COVID dummy (must match training)
    covid_idx = pd.to_datetime(COVID_MONTHS)
    df["is_covid"] = df.index.isin(covid_idx).astype(int)

    # Remove target column if present
    if "TARGET_NFP_CHANGE" in df.columns:
        df = df.drop(columns=["TARGET_NFP_CHANGE"])

    # The most recent complete row is our prediction input
    # (last row = current month, which is what we're forecasting)
    latest_row  = df.iloc[[-1]]
    latest_date = df.index[-1]
    prev_date   = df.index[-2]

    success(f"Latest features row : {latest_date.strftime('%Y-%m-%d')}")
    from dateutil.relativedelta import relativedelta
    next_month = latest_date + relativedelta(months=1)
    info(f"This forecast predicts NFP for : {next_month.strftime('%B %Y')}")

    return df, latest_row, latest_date


# =============================================================================
# STEP 3 — CALCULATE BIAS CORRECTION & INTERVALS
# =============================================================================

def calculate_bias_correction():
    banner("CALCULATING BIAS CORRECTION")

    if not Path(WF_RESULTS_PATH).exists():
        warn("No walk-forward results found — skipping bias correction")
        return 0.0, 150.0, 300.0

    wf = pd.read_csv(WF_RESULTS_PATH, index_col=0, parse_dates=True)
    wf["error"] = wf["actual"] - wf["predicted"]

    covid_idx    = pd.to_datetime(COVID_MONTHS)
    normal_mask  = ~wf.index.isin(covid_idx)

    # Bias correction: rolling mean of recent errors (last 3 years, normal months)
    recent_mask  = (wf.index.year >= 2022) & normal_mask
    recent_errors = wf[recent_mask]["error"]

    # If not enough recent data, use all normal months
    if len(recent_errors) < 12:
        recent_errors = wf[normal_mask]["error"]

    bias_correction = float(recent_errors.mean())

    # Prediction intervals from distribution of normal-month errors
    recent_3yr_mask = (wf.index.year >= 2023) & normal_mask
    recent_errors_interval = wf[recent_3yr_mask]["error"]
    if len(recent_errors_interval) < 6:
        recent_errors_interval = wf[normal_mask]["error"]
    interval_1std = float(recent_errors_interval.std())
    interval_2std = float(recent_errors_interval.std() * 2)

    info(f"Bias correction value : {bias_correction:+.1f}K")
    info(f"Error std dev (1σ)    : ±{interval_1std:.1f}K")
    info(f"Error std dev (2σ)    : ±{interval_2std:.1f}K")
    info(f"Based on {len(recent_errors)} recent normal months")

    return bias_correction, interval_1std, interval_2std


# =============================================================================
# STEP 4 — GENERATE FORECAST
# =============================================================================

def generate_forecast(model, imputer, latest_row):
    banner("GENERATING FORECAST")

    X_imp = imputer.transform(latest_row)
    raw_prediction = float(model.predict(X_imp)[0])

    success(f"Raw model output: {raw_prediction:+,.0f}K jobs")
    return raw_prediction


# =============================================================================
# STEP 5 — APPLY BIAS CORRECTION & BUILD OUTPUT
# =============================================================================

def build_forecast_output(
    raw_pred,
    bias_correction,
    interval_1std,
    interval_2std,
    latest_date,
    feature_matrix,
    meta,
):
    # Apply bias correction
    corrected_pred = raw_pred + bias_correction

    # Confidence intervals around corrected prediction
    optimistic   = corrected_pred + interval_1std
    pessimistic  = corrected_pred - interval_1std
    bull_extreme = corrected_pred + interval_2std
    bear_extreme = corrected_pred - interval_2std

    # Confidence level based on how extreme the forecast is
    wf_normal_mae = meta["metrics"].get("mae_normal", 144)
    deviation_from_trend = abs(corrected_pred - 150)  # 150K is rough long-run avg

    if deviation_from_trend < 50:
        confidence = "HIGH"
        conf_color = Fore.GREEN
    elif deviation_from_trend < 120:
        confidence = "MEDIUM"
        conf_color = Fore.YELLOW
    else:
        confidence = "LOW — Forecast is far from historical average"
        conf_color = Fore.RED

    # Previous month's actual (for context)
    prev_nfp = None
    if "payems_change_lag1" in feature_matrix.columns:
        prev_nfp = float(feature_matrix["payems_change_lag1"].iloc[-1])

    # 6-month average (for context)
    avg_6m = None
    if "payems_6m_avg_change" in feature_matrix.columns:
        avg_6m = float(feature_matrix["payems_6m_avg_change"].iloc[-1])

    from dateutil.relativedelta import relativedelta
    next_month = latest_date + relativedelta(months=1)

    return {
        "forecast_date"  : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "for_month"      : next_month.strftime("%B %Y"),
        "raw_pred"       : round(raw_pred, 0),
        "bias_correction": round(bias_correction, 0),
        "corrected_pred" : round(corrected_pred, 0),
        "optimistic"     : round(optimistic, 0),
        "pessimistic"    : round(pessimistic, 0),
        "bull_extreme"   : round(bull_extreme, 0),
        "bear_extreme"   : round(bear_extreme, 0),
        "confidence"     : confidence,
        "conf_color"     : conf_color,
        "prev_nfp"       : round(prev_nfp, 0) if prev_nfp else None,
        "avg_6m"         : round(avg_6m, 0)   if avg_6m  else None,
    }


# =============================================================================
# STEP 6 — PRINT FORECAST
# =============================================================================

def print_forecast(fc):
    c = Fore.CYAN
    g = Fore.GREEN
    w = Fore.WHITE
    y = Fore.YELLOW

    print()
    print(c + "╔" + "═" * 56 + "╗")
    print(c + "║" + w + f"  NFP FORECAST — {fc['for_month']:<40}" + c + "║")
    print(c + "╠" + "═" * 56 + "╣")

    # Core forecast
    print(c + "║" + w + f"  {'Raw model output':<30} {fc['raw_pred']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "║" + y + f"  {'Bias correction':<30} {fc['bias_correction']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "║" + "─" * 56 + "║")
    print(c + "║" + g + f"  {'BASE FORECAST':<30} {fc['corrected_pred']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "╠" + "═" * 56 + "╣")

    # Confidence range
    print(c + "║" + w + "  CONFIDENCE RANGE:                                    ║")
    print(c + "║" + w + f"  {'Bull case  (+1σ)':<30} {fc['optimistic']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "║" + g + f"  {'Base case':<30} {fc['corrected_pred']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "║" + w + f"  {'Bear case  (-1σ)':<30} {fc['pessimistic']:>+10,.0f}K jobs" + c + "  ║")
    print(c + "╠" + "═" * 56 + "╣")

    # Context
    print(c + "║" + w + "  CONTEXT:                                             ║")
    if fc["prev_nfp"] is not None:
        print(c + "║" + w + f"  {'Previous month NFP':<30} {fc['prev_nfp']:>+10,.0f}K jobs" + c + "  ║")
    if fc["avg_6m"] is not None:
        print(c + "║" + w + f"  {'6-month average':<30} {fc['avg_6m']:>+10,.0f}K jobs" + c + "  ║")

    # Confidence
    print(c + "╠" + "═" * 56 + "╣")
    print(c + "║" + fc["conf_color"] + f"  {'Model Confidence':<30} {fc['confidence']:<22}" + c + "║")
    print(c + "║" + w + f"  {'Forecast generated':<30} {fc['forecast_date']:<22}" + c + "║")
    print(c + "╚" + "═" * 56 + "╝")
    print()

    # Plain text summary for quick reading
    print(Fore.GREEN + f"  BOTTOM LINE: The model expects "
          f"{fc['corrected_pred']:+,.0f}K jobs in {fc['for_month']}.")
    print(Fore.WHITE + f"  Range: {fc['pessimistic']:+,.0f}K (bear) "
          f"to {fc['optimistic']:+,.0f}K (bull)")
    print()


# =============================================================================
# STEP 7 — SAVE TO HISTORY
# =============================================================================

def save_to_history(fc):
    history_path = Path(HISTORY_PATH)

    record = {
        "forecast_date"  : fc["forecast_date"],
        "for_month"      : fc["for_month"],
        "raw_pred"       : fc["raw_pred"],
        "bias_correction": fc["bias_correction"],
        "corrected_pred" : fc["corrected_pred"],
        "optimistic"     : fc["optimistic"],
        "pessimistic"    : fc["pessimistic"],
        "confidence"     : fc["confidence"],
        "actual_nfp"     : None,    # fill this in manually after NFP release
        "error"          : None,    # fill this in after release
    }

    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat(
            [history, pd.DataFrame([record])],
            ignore_index=True
        )
    else:
        history = pd.DataFrame([record])

    history.to_csv(HISTORY_PATH, index=False)
    success(f"Forecast saved to history → {HISTORY_PATH}")
    info("After NFP releases, update 'actual_nfp' column manually to track accuracy")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("NFP FORECASTING SYSTEM — LIVE NOWCASTER", Fore.GREEN)
    info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load everything
    model, imputer, meta = load_model()
    feature_matrix, latest_row, latest_date = load_latest_features()
    bias_correction, interval_1std, interval_2std = calculate_bias_correction()

    # Align columns (model expects same features as training)
    expected_cols = meta["feature_names"]
    for col in expected_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    latest_row = latest_row[expected_cols]

    # Generate raw forecast
    raw_pred = generate_forecast(model, imputer, latest_row)

    # Apply corrections and build output
    fc = build_forecast_output(
        raw_pred, bias_correction,
        interval_1std, interval_2std,
        latest_date, feature_matrix, meta,
    )

    # Print and save
    print_forecast(fc)
    save_to_history(fc)

    banner("NOWCASTER COMPLETE", Fore.GREEN)


if __name__ == "__main__":
    main()