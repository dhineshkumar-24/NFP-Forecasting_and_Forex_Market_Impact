"""
NFP FORECASTING SYSTEM — STEP 3: MODEL TRAINER
===============================================
Trains an XGBoost model to forecast the monthly NFP change.
Uses walk-forward validation to simulate real-world forecasting.
Handles COVID outliers properly.
Saves the trained model for live nowcasting.

Outputs:
  - models/nfp_model.pkl          ← trained model (used by nowcaster)
  - models/model_metadata.json    ← training info & metrics
  - data/walk_forward_results.csv ← all predictions vs actuals

Usage:
    python model_trainer.py
"""

import json
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
init(autoreset=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH  = "data/feature_matrix.parquet"
MODEL_OUTPUT_PATH    = "models/nfp_model.pkl"
IMPUTER_OUTPUT_PATH  = "models/imputer.pkl"
METADATA_PATH        = "models/model_metadata.json"
WF_RESULTS_PATH      = "data/walk_forward_results.csv"

# ── Walk-forward settings ─────────────────────────────────────────────────────
MIN_TRAIN_YEARS  = 10        # minimum years of data before first prediction
WF_START_YEAR    = 2011      # first year we start predicting (test from here)

# ── COVID outlier settings ────────────────────────────────────────────────────
COVID_MONTHS = [
    "2020-03-31", "2020-04-30", "2020-05-31",   # crash + initial bounce
    "2020-06-30", "2020-07-31", "2020-08-31",   # recovery distortion
]
WINSOR_THRESHOLD = 1500      # clip target beyond ±1500K (3 std of normal dist)


def banner(text):
    print(Fore.CYAN + "\n" + "=" * 60)
    print(Fore.CYAN + f"  {text}")
    print(Fore.CYAN + "=" * 60)

def success(text): print(Fore.GREEN  + f"  ✓ {text}")
def info(text):    print(Fore.WHITE  + f"  → {text}")
def warn(text):    print(Fore.YELLOW + f"  ⚠ {text}")
def error(text):   print(Fore.RED    + f"  ✗ {text}")


# =============================================================================
# STEP 1 — LOAD & PREPARE DATA
# =============================================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    banner("LOADING FEATURE MATRIX")

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

    # ── Separate features and target ──────────────────────────────────────────
    y = df["TARGET_NFP_CHANGE"].copy()
    X = df.drop(columns=["TARGET_NFP_CHANGE"])

    # ── Add COVID dummy feature to X ──────────────────────────────────────────
    covid_idx = pd.to_datetime(COVID_MONTHS)
    X["is_covid"] = X.index.isin(covid_idx).astype(int)
    success("Added COVID dummy feature")

    # ── Show target distribution before and after winsorization ───────────────
    banner("TARGET VARIABLE ANALYSIS")

    print(f"\n  {'Metric':<30} {'Raw':>12}  {'Winsorized':>12}")
    print(f"  {'-'*58}")

    y_winsor = y.clip(lower=-WINSOR_THRESHOLD, upper=WINSOR_THRESHOLD)

    for label, series in [("Raw", y), ("Winsorized", y_winsor)]:
        pass  # just computing for table below

    print(f"  {'Mean':<30} {y.mean():>+12.1f}  {y_winsor.mean():>+12.1f}")
    print(f"  {'Std Dev':<30} {y.std():>12.1f}  {y_winsor.std():>12.1f}")
    print(f"  {'Min':<30} {y.min():>+12.1f}  {y_winsor.min():>+12.1f}")
    print(f"  {'Max':<30} {y.max():>+12.1f}  {y_winsor.max():>+12.1f}")
    print(f"  {'Median':<30} {y.median():>+12.1f}  {y_winsor.median():>+12.1f}")

    warn(f"Winsorizing target at ±{WINSOR_THRESHOLD}K — clips {(y.abs() > WINSOR_THRESHOLD).sum()} COVID months")

    return X, y_winsor


# =============================================================================
# STEP 2 — WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    banner("WALK-FORWARD VALIDATION")

    info(f"Training starts : 2000-01")
    info(f"Testing starts  : {WF_START_YEAR}-01")
    info(f"Method          : Expanding window (retrain each year)")
    print()

    results       = []
    test_dates    = X[X.index.year >= WF_START_YEAR].index

    # ── XGBoost config ────────────────────────────────────────────────────────
    model_params = {
        "n_estimators"     : 500,
        "learning_rate"    : 0.05,
        "max_depth"        : 4,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.6,
        "min_child_weight" : 5,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    }

    imputer = SimpleImputer(strategy="median")

    prev_year = None
    for pred_date in test_dates:

        # All data BEFORE this date is training data
        train_mask = X.index < pred_date
        X_train    = X[train_mask]
        y_train    = y[train_mask]
        X_test     = X.loc[[pred_date]]

        # Retrain once per year (expanding window)
        current_year = pred_date.year
        if current_year != prev_year:
            # Fit imputer on training data
            X_train_imp = imputer.fit_transform(X_train)

            # Train model
            model = XGBRegressor(**model_params)
            model.fit(
                X_train_imp, y_train,
                verbose=False
            )

            prev_year = current_year
            info(f"  Retrained for {current_year} | "
                 f"Training months: {len(y_train)} | "
                 f"Features: {X_train.shape[1]}")

        # Predict
        X_test_imp = imputer.transform(X_test)
        pred       = model.predict(X_test_imp)[0]
        actual     = y.loc[pred_date]
        error_val  = actual - pred

        results.append({
            "date"      : pred_date,
            "actual"    : actual,
            "predicted" : pred,
            "error"     : error_val,
            "abs_error" : abs(error_val),
        })

    results_df = pd.DataFrame(results).set_index("date")
    return results_df, model, imputer


def walk_forward_no_covid(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    banner("WALK-FORWARD VALIDATION (COVID EXCLUDED)")

    info(f"Training starts : 2000-01")
    info(f"Testing starts  : {WF_START_YEAR}-01")
    info(f"Method          : Expanding window (retrain each year, COVID excluded)")
    print()

    results       = []
    test_dates    = X[X.index.year >= WF_START_YEAR].index

    # ── XGBoost config ────────────────────────────────────────────────────────
    model_params = {
        "n_estimators"     : 500,
        "learning_rate"    : 0.05,
        "max_depth"        : 4,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.6,
        "min_child_weight" : 5,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    }

    imputer = SimpleImputer(strategy="median")

    prev_year = None
    for pred_date in test_dates:

        # All data BEFORE this date is training data
        train_mask = X.index < pred_date
        X_train    = X[train_mask]
        y_train    = y[train_mask]
        
        # Additionally remove COVID months from training data
        covid_mask = ~X_train.index.isin(pd.to_datetime(COVID_MONTHS))
        X_train    = X_train[covid_mask]
        y_train    = y_train[covid_mask]

        X_test     = X.loc[[pred_date]]

        # Retrain once per year (expanding window)
        current_year = pred_date.year
        if current_year != prev_year:
            # Fit imputer on training data
            X_train_imp = imputer.fit_transform(X_train)

            # Train model
            model = XGBRegressor(**model_params)
            model.fit(
                X_train_imp, y_train,
                verbose=False
            )

            prev_year = current_year
            info(f"  Retrained for {current_year} | "
                 f"Training months: {len(y_train)} | "
                 f"Features: {X_train.shape[1]}")

        # Predict
        X_test_imp = imputer.transform(X_test)
        pred       = model.predict(X_test_imp)[0]
        actual     = y.loc[pred_date]
        error_val  = actual - pred

        results.append({
            "date"      : pred_date,
            "actual"    : actual,
            "predicted" : pred,
            "error"     : error_val,
            "abs_error" : abs(error_val),
        })

    results_df = pd.DataFrame(results).set_index("date")
    return results_df, model, imputer


# =============================================================================
# STEP 3 — EVALUATE PERFORMANCE
# =============================================================================

def evaluate(results_df: pd.DataFrame) -> dict:
    banner("WALK-FORWARD VALIDATION RESULTS")

    actual    = results_df["actual"]
    predicted = results_df["predicted"]
    errors    = results_df["error"]

    mae      = mean_absolute_error(actual, predicted)
    rmse     = np.sqrt(mean_squared_error(actual, predicted))
    mape     = (results_df["abs_error"] / (actual.abs() + 1e-8)).median() * 100
    hit_rate = (np.sign(actual) == np.sign(predicted)).mean() * 100
    bias     = errors.mean()

    # Exclude COVID months for a "normal conditions" MAE
    covid_idx    = pd.to_datetime(COVID_MONTHS)
    normal_mask  = ~results_df.index.isin(covid_idx)
    mae_normal   = mean_absolute_error(
        actual[normal_mask], predicted[normal_mask]
    )

    metrics = {
        "mae"         : round(mae, 1),
        "mae_normal"  : round(mae_normal, 1),
        "rmse"        : round(rmse, 1),
        "mape"        : round(mape, 1),
        "hit_rate"    : round(hit_rate, 1),
        "bias"        : round(bias, 1),
        "n_forecasts" : len(results_df),
    }

    print(f"\n  {'Metric':<40} {'Value':>10}")
    print(f"  {'-'*54}")
    print(f"  {'MAE (all months)':<40} {mae:>+10.1f}K")
    print(f"  {'MAE (excluding COVID months)':<40} {mae_normal:>+10.1f}K")
    print(f"  {'RMSE':<40} {rmse:>10.1f}K")
    print(f"  {'Directional Accuracy (hit rate)':<40} {hit_rate:>9.1f}%")
    print(f"  {'Forecast Bias':<40} {bias:>+10.1f}K")
    print(f"  {'Total forecasts made':<40} {len(results_df):>10}")

    print()

    # ── Year-by-year breakdown ─────────────────────────────────────────────────
    print(Fore.CYAN + "  Year-by-Year Performance:")
    print(f"  {'Year':<8} {'MAE':>8} {'Bias':>8} {'Hit%':>8} {'Forecasts':>10}")
    print(f"  {'-'*48}")

    for year, group in results_df.groupby(results_df.index.year):
        yr_mae  = group["abs_error"].mean()
        yr_bias = group["error"].mean()
        yr_hit  = (np.sign(group["actual"]) == np.sign(group["predicted"])).mean() * 100
        print(f"  {year:<8} {yr_mae:>8.1f} {yr_bias:>+8.1f} {yr_hit:>7.1f}% {len(group):>10}")

    return metrics


# =============================================================================
# STEP 4 — TRAIN FINAL MODEL ON ALL DATA
# =============================================================================

def train_final_model(X: pd.DataFrame, y: pd.Series):
    banner("TRAINING FINAL MODEL ON ALL DATA")

    final_imputer = SimpleImputer(strategy="median")
    X_imputed     = final_imputer.fit_transform(X)

    final_model = XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 4,
        subsample         = 0.8,
        colsample_bytree  = 0.6,
        min_child_weight  = 5,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )
    final_model.fit(X_imputed, y, verbose=False)

    success(f"Final model trained on {len(y)} months (2000–{y.index.max().year})")
    return final_model, final_imputer


# =============================================================================
# STEP 5 — FEATURE IMPORTANCE (TOP 20)
# =============================================================================

def feature_importance(model, X: pd.DataFrame, imputer):
    banner("TOP 20 MOST IMPORTANT FEATURES")

    X_imp = imputer.transform(X)
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    top20 = importances.head(20)
    max_imp = top20.max()

    print()
    for i, (feat, imp) in enumerate(top20.items(), 1):
        bar_len = int((imp / max_imp) * 35)
        bar     = "█" * bar_len
        print(f"  {i:>2}. {feat:<40} {bar} {imp:.4f}")

    print()
    info("Full importance saved in model metadata")

    return importances


# =============================================================================
# STEP 6 — SAVE EVERYTHING
# =============================================================================

def save_all(model, imputer, metrics, importances, results_df, X):
    banner("SAVING MODEL & RESULTS")

    Path("models").mkdir(exist_ok=True)

    # Save model
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    success(f"Model saved   → {MODEL_OUTPUT_PATH}")

    # Save imputer
    with open(IMPUTER_OUTPUT_PATH, "wb") as f:
        pickle.dump(imputer, f)
    success(f"Imputer saved → {IMPUTER_OUTPUT_PATH}")

    # Save walk-forward results
    results_df.to_csv(WF_RESULTS_PATH)
    success(f"WF results    → {WF_RESULTS_PATH}")

    # Save metadata
    metadata = {
        "trained_on"    : datetime.now().isoformat(),
        "train_start"   : str(X.index.min().date()),
        "train_end"     : str(X.index.max().date()),
        "n_features"    : X.shape[1],
        "n_months"      : X.shape[0],
        "metrics"       : metrics,
        "top_features"  : importances.head(20).to_dict(),
        "feature_names" : list(X.columns),
        "covid_months"  : COVID_MONTHS,
        "winsor_threshold": WINSOR_THRESHOLD,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    success(f"Metadata      → {METADATA_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("NFP FORECASTING SYSTEM — MODEL TRAINER")
    info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    X, y = load_and_prepare()

    # Walk-forward validation (Model A: COVID included)
    results_df, wf_model_a, wf_imputer_a = walk_forward_validation(X, y)
    metrics_a = evaluate(results_df)

    # Walk-forward validation (Model B: COVID excluded)
    results_df_b, wf_model_b, wf_imputer_b = walk_forward_no_covid(X, y)
    metrics_b = evaluate(results_df_b)

    # Print comparison table
    mae_a_str = f"{metrics_a['mae_normal']:.1f}K"
    mae_b_str = f"{metrics_b['mae_normal']:.1f}K"
    hit_a_str = f"{metrics_a['hit_rate']:.1f}%"
    hit_b_str = f"{metrics_b['hit_rate']:.1f}%"
    bias_a_str = f"{metrics_a['bias']:+.1f}K"
    bias_b_str = f"{metrics_b['bias']:+.1f}K"

    print("\n=== MODEL COMPARISON ===")
    print("                         Model A          Model B")
    print("                    (COVID included) (COVID excluded)")
    print(f"MAE normal months:       {mae_a_str:<16} {mae_b_str:<16}")
    print(f"Directional Accuracy:    {hit_a_str:<16} {hit_b_str:<16}")
    print(f"Bias:                    {bias_a_str:<16} {bias_b_str:<16}")
    print()

    # Determine winner
    if metrics_b['mae_normal'] < metrics_a['mae_normal']:
        print(Fore.GREEN + "Model B (COVID excluded) won due to lower MAE on normal months!")
        winner_name = "Model B (COVID excluded)"
        winner_metrics = metrics_b
        winner_results = results_df_b
        # Exclude COVID months from final training
        covid_mask = ~X.index.isin(pd.to_datetime(COVID_MONTHS))
        X_final = X[covid_mask]
        y_final = y[covid_mask]
    else:
        print(Fore.GREEN + "Model A (COVID included) won due to lower MAE on normal months!")
        winner_name = "Model A (COVID included)"
        winner_metrics = metrics_a
        winner_results = results_df
        X_final = X
        y_final = y

    # Train final model on all data (using winner's training data)
    final_model, final_imputer = train_final_model(X_final, y_final)

    # Feature importance
    importances = feature_importance(final_model, X_final, final_imputer)

    # Save everything
    save_all(final_model, final_imputer, winner_metrics, importances, winner_results, X_final)

    banner("STEP 3 COMPLETE — MODEL TRAINED & SAVED")
    print(Fore.GREEN + f"""
  Your winning model ({winner_name}) is ready. Key metrics:
  ├─ MAE (normal months)  : ±{winner_metrics['mae_normal']:.0f}K jobs
  ├─ Directional Accuracy : {winner_metrics['hit_rate']:.1f}%
  ├─ Forecast Bias        : {winner_metrics['bias']:+.1f}K
  └─ Total WF forecasts   : {winner_metrics['n_forecasts']}

  Next step → python nowcaster.py
    """)


if __name__ == "__main__":
    main()
