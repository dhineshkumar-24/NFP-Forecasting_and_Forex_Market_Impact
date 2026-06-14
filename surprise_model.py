"""
NFP FORECASTING SYSTEM — STAGE 2: SURPRISE MODEL
=================================================
Predicts whether the upcoming NFP will BEAT or MISS consensus,
and estimates the magnitude of the surprise.

NFP Surprise = Actual NFP − Consensus Estimate

This is what actually moves markets. A +180K print means nothing
without knowing consensus was +200K (a miss) or +150K (a beat).

Outputs per forecast:
  - Beat probability      (e.g. 68%)
  - Miss probability      (e.g. 32%)
  - Expected surprise     (e.g. +42K)
  - Signal                (BEAT / MISS / INLINE)
  - Confidence            (HIGH / MEDIUM / LOW)

Usage:
    python surprise_model.py

Run AFTER:
    python data_collector.py
    python feature_engineer.py
    python nowcaster.py        ← need Stage 1 forecast first
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
init(autoreset=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
CONSENSUS_PATH      = "data/consensus_data.csv"
FEATURE_MATRIX_PATH = "data/feature_matrix.parquet"
WF_RESULTS_PATH     = "data/walk_forward_results.csv"
FORECAST_HISTORY    = "data/forecast_history.csv"

CLASSIFIER_PATH     = "models/surprise_classifier.pkl"
REGRESSOR_PATH      = "models/surprise_regressor.pkl"
S2_IMPUTER_PATH     = "models/surprise_imputer.pkl"
S2_METADATA_PATH    = "models/surprise_metadata.json"
S2_RESULTS_PATH     = "data/surprise_wf_results.csv"

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
# STEP 1 — LOAD & MERGE DATA
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    banner("LOADING & MERGING DATA")

    # ── Consensus data ─────────────────────────────────────────────────────
    consensus = pd.read_csv(CONSENSUS_PATH, parse_dates=["date"])
    consensus = consensus.set_index("date")
    consensus.index = pd.to_datetime(consensus.index)
    consensus.sort_index(inplace=True)
    success(f"Consensus data: {len(consensus)} months")

    # ── Feature matrix from Stage 1 ────────────────────────────────────────
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features.index = pd.to_datetime(features.index)
    features.sort_index(inplace=True)

    # Drop target from feature matrix (not a feature here)
    if "TARGET_NFP_CHANGE" in features.columns:
        features = features.drop(columns=["TARGET_NFP_CHANGE"])
    success(f"Feature matrix: {features.shape[0]} rows × {features.shape[1]} cols")

    # ── Stage 1 walk-forward predictions ───────────────────────────────────
    if Path(WF_RESULTS_PATH).exists():
        wf = pd.read_csv(WF_RESULTS_PATH, index_col=0, parse_dates=True)
        wf.index = pd.to_datetime(wf.index)
        wf = wf.rename(columns={
            "predicted" : "stage1_forecast",
            "actual"    : "stage1_actual",
        })
        features = features.join(wf[["stage1_forecast"]], how="left")
        success(f"Stage 1 forecasts merged: {wf['stage1_forecast'].notna().sum()} months")
    else:
        warn("No Stage 1 walk-forward results found — stage1_forecast will be empty")
        features["stage1_forecast"] = np.nan

    # ── Merge everything ───────────────────────────────────────────────────
    df = features.join(consensus, how="inner")

    info(f"Merged dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

    return df


# =============================================================================
# STEP 2 — ENGINEER SURPRISE-SPECIFIC FEATURES
# =============================================================================

def engineer_surprise_features(df: pd.DataFrame) -> pd.DataFrame:
    banner("ENGINEERING SURPRISE FEATURES")

    # ── Core surprise signal: Stage 1 model vs consensus ───────────────────
    if "stage1_forecast" in df.columns:
        df["model_vs_consensus"]     = df["stage1_forecast"] - df["consensus_forecast"]
        df["model_above_consensus"]  = (df["model_vs_consensus"] > 0).astype(int)
        success("Model vs consensus gap (primary alpha signal)")

    # ── Consensus accuracy history ──────────────────────────────────────────
    df["consensus_error"]     = df["actual_nfp"] - df["consensus_forecast"]
    df["consensus_err_lag1"]  = df["consensus_error"].shift(1)
    df["consensus_err_lag2"]  = df["consensus_error"].shift(2)
    df["consensus_err_lag3"]  = df["consensus_error"].shift(3)

    # Rolling consensus bias (has consensus been consistently under/over?)
    df["consensus_bias_3m"]   = df["consensus_error"].shift(1).rolling(3).mean()
    df["consensus_bias_6m"]   = df["consensus_error"].shift(1).rolling(6).mean()
    df["consensus_bias_12m"]  = df["consensus_error"].shift(1).rolling(12).mean()
    success("Consensus historical accuracy & bias features")

    # ── Consensus revision momentum ─────────────────────────────────────────
    df["consensus_mom"]       = df["consensus_forecast"].diff(1).shift(1)
    df["consensus_3m_change"] = df["consensus_forecast"].diff(3).shift(1)
    success("Consensus revision momentum")

    # ── Historical beat/miss streaks ────────────────────────────────────────
    beat_binary = (df["actual_nfp"] > df["consensus_forecast"]).astype(int).shift(1)
    df["beat_streak"] = beat_binary.groupby(
        (beat_binary != beat_binary.shift()).cumsum()
    ).cumcount() + 1
    df["beat_streak"] = df["beat_streak"] * beat_binary  # zero out on misses

    df["beats_last_3m"] = beat_binary.rolling(3).sum()
    df["beats_last_6m"] = beat_binary.rolling(6).sum()
    success("Beat/miss streak and recent hit rate features")

    # ── Surprise magnitude history ──────────────────────────────────────────
    df["surprise_lag1"]    = df["surprise"].shift(1)
    df["surprise_lag2"]    = df["surprise"].shift(2)
    df["surprise_lag3"]    = df["surprise"].shift(3)
    df["surprise_abs_avg"] = df["surprise"].shift(1).abs().rolling(6).mean()
    success("Historical surprise magnitude features")

    # ── Beat/miss target columns ────────────────────────────────────────────
    # Classification target: 1 = BEAT, 0 = MISS/INLINE
    df["TARGET_BEAT"]      = (df["surprise"] > 0).astype(int)

    # Regression target: surprise magnitude in K jobs
    df["TARGET_SURPRISE"]  = df["surprise"]

    info(f"Beat rate in dataset: {df['TARGET_BEAT'].mean()*100:.1f}%")
    info(f"Avg surprise magnitude: {df['TARGET_SURPRISE'].mean():+.1f}K")
    info(f"Surprise std dev: {df['TARGET_SURPRISE'].std():.1f}K")

    return df


# =============================================================================
# STEP 3 — SELECT FEATURES FOR STAGE 2
# =============================================================================

def select_features(df: pd.DataFrame) -> tuple[list, list]:
    """
    Stage 2 uses a focused subset of features.
    Too many features with limited data → overfitting.
    We keep the most economically meaningful ones.
    """

    # Surprise-specific features (always include)
    surprise_features = [
        "model_vs_consensus", "model_above_consensus",
        "consensus_err_lag1", "consensus_err_lag2", "consensus_err_lag3",
        "consensus_bias_3m", "consensus_bias_6m", "consensus_bias_12m",
        "consensus_mom", "consensus_3m_change",
        "beat_streak", "beats_last_3m", "beats_last_6m",
        "surprise_lag1", "surprise_lag2", "surprise_lag3",
        "surprise_abs_avg",
        "stage1_forecast",
    ]

    # Best macro features from Stage 1 importance analysis
    macro_features = [
        "UNRATE_mom3", "UNRATE_zscore",
        "ICSA_lag1", "ICSA_lag2", "ICSA_lag3",
        "CCSA_mom3", "CCSA_zscore",
        "AWHAETP_ma3", "AWHAETP_ma6", "AWHAETP_lag2",
        "CIVPART_ma12", "CIVPART_lag3",
        "ADPMNUSNERSA_lag1", "adp_change_lag1",
        "payems_change_lag1", "payems_3m_avg_change",
        "INDPRO_zscore", "TCU_zscore",
        "NFCI_lag1", "NFCI_zscore",
        "VIXCLS_lag1", "VIXCLS_zscore",
        "DGS10_lag1", "T10Y2Y_lag1",
        "month_sin", "month_cos", "month", "quarter",
        "is_recession", "rate_hiking", "rate_cutting",
        "real_rate", "claims_trend", "claims_ratio",
    ]

    # Filter to only columns that exist in df
    all_features = surprise_features + macro_features
    available    = [f for f in all_features if f in df.columns]
    excluded     = [f for f in all_features if f not in df.columns]

    if excluded:
        warn(f"Missing {len(excluded)} features (not in dataset): {excluded[:5]}...")

    success(f"Using {len(available)} features for Stage 2")
    return available, surprise_features


# =============================================================================
# STEP 4 — WALK-FORWARD VALIDATION (STAGE 2)
# =============================================================================

def walk_forward_stage2(
    df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    banner("STAGE 2 WALK-FORWARD VALIDATION")

    # Remove COVID months
    covid_idx  = pd.to_datetime(COVID_MONTHS)
    df_clean   = df[~df.index.isin(covid_idx)].copy()

    # Only use rows where we have both features and targets
    df_clean   = df_clean.dropna(subset=["TARGET_BEAT", "TARGET_SURPRISE"])

    # Walk-forward starts at 2018 (need enough surprise history)
    wf_start   = 2018
    test_dates = df_clean[df_clean.index.year >= wf_start].index

    info(f"Training starts : 2015-01")
    info(f"Testing starts  : {wf_start}-01")
    info(f"Test months     : {len(test_dates)}")

    results    = []
    imputer    = SimpleImputer(strategy="median")
    prev_year  = None
    clf = None
    reg = None

    for pred_date in test_dates:
        train_mask = df_clean.index < pred_date
        X_train    = df_clean[train_mask][feature_cols]
        y_clf      = df_clean[train_mask]["TARGET_BEAT"]
        y_reg      = df_clean[train_mask]["TARGET_SURPRISE"]
        X_test     = df_clean.loc[[pred_date], feature_cols]

        if len(X_train) < 24:
            continue

        # Retrain once per year
        curr_year = pred_date.year
        if curr_year != prev_year:
            X_tr_imp = imputer.fit_transform(X_train)

            # Classifier: BEAT vs MISS
            clf = XGBClassifier(
                n_estimators     = 300,
                learning_rate    = 0.05,
                max_depth        = 3,
                subsample        = 0.8,
                colsample_bytree = 0.6,
                min_child_weight = 3,
                reg_alpha        = 0.5,
                random_state     = 42,
                n_jobs           = -1,
                verbosity        = 0,
                eval_metric      = "logloss",
            )
            clf.fit(X_tr_imp, y_clf)

            # Regressor: surprise magnitude
            reg = XGBRegressor(
                n_estimators     = 300,
                learning_rate    = 0.05,
                max_depth        = 3,
                subsample        = 0.8,
                colsample_bytree = 0.6,
                min_child_weight = 3,
                reg_alpha        = 0.5,
                random_state     = 42,
                n_jobs           = -1,
                verbosity        = 0,
            )
            reg.fit(X_tr_imp, y_reg)

            prev_year = curr_year
            info(f"  Retrained {curr_year} | Train months: {len(y_clf)}")

        # Predict
        X_te_imp       = imputer.transform(X_test)
        beat_prob      = float(clf.predict_proba(X_te_imp)[0][1])
        miss_prob      = 1 - beat_prob
        pred_surprise  = float(reg.predict(X_te_imp)[0])
        pred_direction = 1 if beat_prob >= 0.5 else 0

        actual_beat    = int(df_clean.loc[pred_date, "TARGET_BEAT"])
        actual_surp    = float(df_clean.loc[pred_date, "TARGET_SURPRISE"])
        consensus      = float(df_clean.loc[pred_date, "consensus_forecast"])
        actual_nfp     = float(df_clean.loc[pred_date, "actual_nfp"])

        results.append({
            "date"           : pred_date,
            "consensus"      : consensus,
            "actual_nfp"     : actual_nfp,
            "actual_surprise": actual_surp,
            "beat_prob"      : round(beat_prob, 3),
            "miss_prob"      : round(miss_prob, 3),
            "pred_surprise"  : round(pred_surprise, 0),
            "pred_direction" : pred_direction,
            "actual_beat"    : actual_beat,
            "correct"        : int(pred_direction == actual_beat),
        })

    return pd.DataFrame(results).set_index("date"), clf, reg, imputer


# =============================================================================
# STEP 5 — EVALUATE STAGE 2
# =============================================================================

def evaluate_stage2(results: pd.DataFrame) -> dict:
    banner("STAGE 2 RESULTS")

    acc      = accuracy_score(results["actual_beat"], results["pred_direction"])
    surp_mae = mean_absolute_error(results["actual_surprise"], results["pred_surprise"])

    # High-confidence predictions (beat_prob > 0.65 or < 0.35)
    high_conf = results[
        (results["beat_prob"] > 0.65) | (results["beat_prob"] < 0.35)
    ]
    hc_acc = accuracy_score(
        high_conf["actual_beat"], high_conf["pred_direction"]
    ) if len(high_conf) > 0 else 0

    # Beat rate in test set
    beat_rate = results["actual_beat"].mean()

    metrics = {
        "direction_accuracy"   : round(acc * 100, 1),
        "high_conf_accuracy"   : round(hc_acc * 100, 1),
        "high_conf_signals"    : len(high_conf),
        "surprise_mae"         : round(surp_mae, 1),
        "total_forecasts"      : len(results),
        "actual_beat_rate"     : round(beat_rate * 100, 1),
    }

    print(f"\n  {'Metric':<40} {'Value':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Direction Accuracy (all)':<40} {acc*100:>11.1f}%")
    print(f"  {'Direction Accuracy (high confidence)':<40} {hc_acc*100:>11.1f}%")
    print(f"  {'High confidence signals':<40} {len(high_conf):>12}")
    print(f"  {'Surprise MAE':<40} {surp_mae:>+11.1f}K")
    print(f"  {'Actual beat rate in test set':<40} {beat_rate*100:>11.1f}%")
    print(f"  {'Total forecasts':<40} {len(results):>12}")

    # Year by year
    print(Fore.CYAN + "\n  Year-by-Year Direction Accuracy:")
    print(f"  {'Year':<8} {'Acc%':>8} {'BeatRate':>10} {'HiConf':>8} {'N':>6}")
    print(f"  {'-'*44}")

    for year, grp in results.groupby(results.index.year):
        yr_acc  = grp["correct"].mean() * 100
        yr_beat = grp["actual_beat"].mean() * 100
        yr_hc   = len(grp[(grp["beat_prob"] > 0.65) | (grp["beat_prob"] < 0.35)])
        print(f"  {year:<8} {yr_acc:>7.1f}% {yr_beat:>9.1f}% {yr_hc:>8} {len(grp):>6}")

    # Show some example predictions
    print(Fore.CYAN + "\n  Sample Predictions (last 12 months):")
    print(f"  {'Date':<12} {'Cons':>8} {'Actual':>8} {'Surp':>8} "
          f"{'BeatP':>7} {'Signal':>8} {'Correct':>8}")
    print(f"  {'-'*65}")

    for dt, row in results.tail(12).iterrows():
        signal  = "BEAT" if row["beat_prob"] >= 0.5 else "MISS"
        correct = "✓" if row["correct"] else "✗"
        print(
            f"  {str(dt.date()):<12} "
            f"{row['consensus']:>7,.0f} "
            f"{row['actual_nfp']:>8,.0f} "
            f"{row['actual_surprise']:>+8,.0f} "
            f"{row['beat_prob']:>7.1%} "
            f"{signal:>8} "
            f"{correct:>8}"
        )

    return metrics


# =============================================================================
# STEP 6 — TRAIN FINAL MODELS
# =============================================================================

def train_final_models(df, feature_cols):
    banner("TRAINING FINAL STAGE 2 MODELS")

    covid_idx = pd.to_datetime(COVID_MONTHS)
    df_clean  = df[~df.index.isin(covid_idx)].copy()
    df_clean  = df_clean.dropna(subset=["TARGET_BEAT", "TARGET_SURPRISE"])

    X = df_clean[feature_cols]
    y_clf = df_clean["TARGET_BEAT"]
    y_reg = df_clean["TARGET_SURPRISE"]

    imputer  = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
        reg_alpha=0.5, random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="logloss",
    )
    clf.fit(X_imputed, y_clf)

    reg = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
        reg_alpha=0.5, random_state=42, n_jobs=-1, verbosity=0,
    )
    reg.fit(X_imputed, y_reg)

    success(f"Final models trained on {len(y_clf)} months")
    return clf, reg, imputer


# =============================================================================
# STEP 7 — LIVE PREDICTION
# =============================================================================

def live_surprise_prediction(
    clf, reg, imputer,
    df, feature_cols,
    next_consensus: float,
):
    banner("LIVE SURPRISE FORECAST", Fore.GREEN)

    # Get latest feature row
    latest = df.iloc[[-1]][feature_cols].copy()

    # Inject the consensus estimate for the upcoming month
    if "consensus_forecast" in latest.columns:
        latest["consensus_forecast"] = next_consensus

    # Recompute model_vs_consensus with the new consensus
    if "stage1_forecast" in df.columns and "model_vs_consensus" in latest.columns:
        stage1 = float(df["stage1_forecast"].iloc[-1]) if not pd.isna(
            df["stage1_forecast"].iloc[-1]) else 0
        latest["model_vs_consensus"]    = stage1 - next_consensus
        latest["model_above_consensus"] = 1 if stage1 > next_consensus else 0

    X_imp      = imputer.transform(latest)
    beat_prob  = float(clf.predict_proba(X_imp)[0][1])
    miss_prob  = 1 - beat_prob
    pred_surp  = float(reg.predict(X_imp)[0])
    signal     = "BEAT" if beat_prob >= 0.5 else "MISS"

    if beat_prob > 0.65 or beat_prob < 0.35:
        confidence = "HIGH"
        conf_color = Fore.GREEN
    elif beat_prob > 0.55 or beat_prob < 0.45:
        confidence = "MEDIUM"
        conf_color = Fore.YELLOW
    else:
        confidence = "LOW"
        conf_color = Fore.RED

    # Load Stage 1 forecast for display
    stage1_forecast = None
    if Path(FORECAST_HISTORY).exists():
        hist = pd.read_csv(FORECAST_HISTORY)
        if not hist.empty:
            stage1_forecast = hist["corrected_pred"].iloc[-1]

    # Print forecast box
    c = Fore.CYAN
    w = Fore.WHITE
    g = Fore.GREEN
    y = Fore.YELLOW

    print()
    print(c + "╔" + "═" * 56 + "╗")
    print(c + "║" + w + f"  NFP SURPRISE FORECAST (Stage 2)               " + c + "  ║")
    print(c + "╠" + "═" * 56 + "╣")
    print(c + "║" + w + f"  {'Market consensus':<30} {next_consensus:>+10,.0f} jobs" + c + "  ║")
    if stage1_forecast:
        stage1_display = stage1_forecast * 1000 if stage1_forecast < 1000 else stage1_forecast
        print(c + "║" + w + f"  {'Our Stage 1 forecast':<30} {stage1_display:>+10,.0f} jobs" + c + "  ║")
        gap = stage1_display - (next_consensus * 1000)
        print(c + "║" + y + f"  {'Model vs consensus gap':<30} {gap:>+10,.0f} jobs" + c + "  ║")
    print(c + "╠" + "═" * 56 + "╣")
    print(c + "║" + w + f"  {'Predicted surprise':<30} {pred_surp:>+10,.0f} jobs" + c + "  ║")
    print(c + "╠" + "═" * 56 + "╣")
    print(c + "║" + w + f"  {'Beat probability':<30} {beat_prob:>10.1%}" + c + "          ║")
    print(c + "║" + w + f"  {'Miss probability':<30} {miss_prob:>10.1%}" + c + "          ║")
    print(c + "╠" + "═" * 56 + "╣")
    print(c + "║" + conf_color + f"  {'SIGNAL':<30} {signal:<24}" + c + "║")
    print(c + "║" + conf_color + f"  {'Confidence':<30} {confidence:<24}" + c + "║")
    print(c + "╚" + "═" * 56 + "╝")
    print()

    return {
        "beat_probability" : round(beat_prob, 3),
        "miss_probability" : round(miss_prob, 3),
        "predicted_surprise": round(pred_surp, 0),
        "signal"           : signal,
        "confidence"       : confidence,
    }


# =============================================================================
# STEP 8 — SAVE MODELS
# =============================================================================

def save_models(clf, reg, imputer, metrics, feature_cols):
    banner("SAVING STAGE 2 MODELS")

    Path("models").mkdir(exist_ok=True)

    with open(CLASSIFIER_PATH, "wb") as f: pickle.dump(clf,     f)
    with open(REGRESSOR_PATH,  "wb") as f: pickle.dump(reg,     f)
    with open(S2_IMPUTER_PATH, "wb") as f: pickle.dump(imputer, f)

    success(f"Classifier → {CLASSIFIER_PATH}")
    success(f"Regressor  → {REGRESSOR_PATH}")

    metadata = {
        "trained_on"      : datetime.now().isoformat(),
        "metrics"         : metrics,
        "feature_cols"    : feature_cols,
        "n_features"      : len(feature_cols),
    }
    with open(S2_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    success(f"Metadata   → {S2_METADATA_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("NFP FORECASTING SYSTEM — STAGE 2: SURPRISE MODEL")
    info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and merge
    df = load_and_merge()

    # Engineer surprise features
    df = engineer_surprise_features(df)

    # Select features
    feature_cols, _ = select_features(df)

    # Walk-forward validation
    results, wf_clf, wf_reg, wf_imputer = walk_forward_stage2(df, feature_cols)

    # Evaluate
    metrics = evaluate_stage2(results)
    results.to_csv(S2_RESULTS_PATH)
    success(f"Walk-forward results → {S2_RESULTS_PATH}")

    # Train final models
    clf, reg, imputer = train_final_models(df, feature_cols)

    # Save
    save_models(clf, reg, imputer, metrics, feature_cols)

    # ── Live prediction for next NFP ──────────────────────────────────────
    banner("ENTER CONSENSUS FOR UPCOMING NFP")
    print(Fore.YELLOW + """
  To generate the live surprise forecast, enter the current
  market consensus estimate for the upcoming NFP release.

  Where to find it:
    → https://www.forexfactory.com (Economic Calendar)
    → https://investing.com/economic-calendar/
    → Google: "NFP consensus forecast [current month]"
    """)

    try:
        consensus_input = input(
            Fore.CYAN + "  Enter consensus estimate in thousands (e.g. 130 for 130K jobs): "
        )
        raw_val = float(consensus_input.strip())
        # Accept both formats: 130 and 130000
        next_consensus = raw_val / 1000 if raw_val > 10000 else raw_val
        live_surprise_prediction(clf, reg, imputer, df, feature_cols, next_consensus)
    except (ValueError, KeyboardInterrupt):
        warn("No consensus entered — skipping live prediction")
        info("Run nowcaster.py after entering consensus to get surprise forecast")

    banner("STAGE 2 COMPLETE", Fore.GREEN)
    print(Fore.GREEN + f"""
  Stage 2 is ready. Key metrics:
  ├─ Direction Accuracy      : {metrics['direction_accuracy']:.1f}%
  ├─ High-Conf Accuracy      : {metrics['high_conf_accuracy']:.1f}%
  ├─ High-Conf Signals       : {metrics['high_conf_signals']} months
  └─ Surprise MAE            : ±{metrics['surprise_mae']:.0f}K

  Next step → Stage 3: Market reaction model
    """)


if __name__ == "__main__":
    main()