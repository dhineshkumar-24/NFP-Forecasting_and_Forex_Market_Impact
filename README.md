# NFP Forecasting System

> **Institutional-grade macroeconomic nowcasting for US Non-Farm Payrolls**  
> A 3-stage ML pipeline: Level Forecast → Surprise Signal → Market Reaction

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Monthly Workflow](#monthly-workflow)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — NFP Level Forecast](#stage-1--nfp-level-forecast)
  - [Stage 2 — Surprise Signal](#stage-2--surprise-signal)
  - [Stage 3 — Market Reaction](#stage-3--market-reaction)
- [Dashboard](#dashboard)
- [Model Performance](#model-performance)
- [Data Sources](#data-sources)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

---

## Overview

This system collects 39 macroeconomic indicators from the FRED API, engineers 469 predictive features, and trains an XGBoost model using expanding-window walk-forward validation to produce live NFP nowcasts before every monthly release.

**What it does:**
- Downloads and preprocesses 39 FRED macroeconomic series
- Engineers 469 features per month (lags, rolling stats, z-scores, cross-series spreads)
- Trains an XGBoost regressor with COVID-aware walk-forward validation
- Predicts NFP level, beat/miss signal, and historical market reaction patterns
- Exports a self-contained interactive HTML dashboard

**Key results (out-of-sample, 2011–2026):**
| Metric | Value |
|---|---|
| MAE (excluding COVID) | ±144K jobs |
| Directional Accuracy | 90.8% |
| Top feature | `is_recession` |
| Training span | 311 months (2000–2026) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NFP FORECASTING PIPELINE                 │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  STEP 1      │  STEP 2      │  STEP 3      │  STEP 4       │
│  Data        │  Feature     │  Model       │  Nowcaster    │
│  Collector   │  Engineer    │  Trainer     │               │
│              │              │              │               │
│  39 FRED     │  469         │  XGBoost     │  Bias-        │
│  series →    │  features →  │  walk-fwd →  │  corrected    │
│  parquet     │  matrix      │  model.pkl   │  forecast     │
└──────────────┴──────────────┴──────────────┴───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Stage 2          Stage 3         Dashboard
           Surprise Model   Market Reaction   Streamlit /
           Beat vs Miss     Gold, FX, S&P     HTML Report
```

---

## Project Structure

```
nfp_forecaster/
│
├── config.py                       # API keys, series dictionary, hyperparameters
├── data_collector.py               # Step 1: Fetch & merge 39 FRED indicators
├── feature_engineer.py             # Step 2: Engineer 469 predictive features
├── model_trainer.py                # Step 3: Walk-forward validation + model training
├── nowcaster.py                    # Step 4: Live NFP nowcast generation
│
├── surprise_model.py               # Stage 2: Beat/Miss classifier + surprise regressor
├── market_reaction_model.py        # Stage 3: Historical market reaction by surprise bucket
│
├── run_forecast.py                 # Master runner — executes full pipeline end-to-end
├── dashboard.py                    # Streamlit dashboard
│
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # Per-series parquet files from FRED
│   ├── processed_macro_data.parquet  # Wide month-end merged dataset
│   ├── feature_matrix.parquet      # 469-feature training matrix
│   ├── walk_forward_results.csv    # Stage 1 OOS predictions vs actuals
│   ├── surprise_wf_results.csv     # Stage 2 OOS beat/miss predictions
│   ├── market_reaction_results.csv # Stage 3 bucket statistics
│   ├── consensus_data.csv          # Historical consensus + actual NFP + surprise
│   └── forecast_history.csv        # Live forecast run log
│
├── models/
│   ├── nfp_model.pkl               # Trained XGBoost regressor (Stage 1)
│   ├── imputer.pkl                 # SimpleImputer for Stage 1
│   ├── surprise_classifier.pkl     # XGBoost beat/miss classifier (Stage 2)
│   ├── surprise_regressor.pkl      # XGBoost surprise magnitude regressor (Stage 2)
│   ├── surprise_imputer.pkl        # SimpleImputer for Stage 2
│   ├── model_metadata.json         # Stage 1 metrics, feature importances
│   └── surprise_metadata.json      # Stage 2 metrics and feature list
│
└── reports/
    └── nfp_forecast_report.html    # Auto-generated institutional HTML dashboard
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
# Create
python -m venv nfp_env

# Activate — macOS/Linux
source nfp_env/bin/activate

# Activate — Windows (PowerShell)
nfp_env\Scripts\activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your FRED API key

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_actual_api_key_here
```

Get a free API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

> **No API key?** Run with `--mock` to generate synthetic data for testing:
> ```bash
> python data_collector.py --mock
> ```

### 4. Run the full pipeline

```bash
python run_forecast.py
```

This runs all 5 steps automatically, prompts for the market consensus estimate, and generates the HTML report.

---

## Monthly Workflow

Run this **every month on Thursday before NFP Friday**, after the ADP report is released (typically Wednesday morning). This timing ensures the most recent jobless claims, ADP payroll data, and month-end financial readings are captured.

```bash
# Full pipeline (recommended)
python run_forecast.py

# Or step-by-step
python data_collector.py      # Fetch latest FRED data
python feature_engineer.py    # Rebuild feature matrix
python model_trainer.py       # Retrain model (optional — monthly)
python nowcaster.py            # Generate Stage 1 forecast
python surprise_model.py       # Generate Stage 2 surprise signal
python market_reaction_model.py  # Generate Stage 3 market patterns
```

**After the NFP release on Friday:**  
Open `data/forecast_history.csv` and fill in the `actual_nfp` and `error` columns to track accuracy over time.

---

## Pipeline Stages

### Stage 1 — NFP Level Forecast

**Script:** `nowcaster.py`  
**Model:** XGBoost Regressor (trained in `model_trainer.py`)

Predicts the raw number of jobs added/lost in the upcoming month.

**Feature groups used:**
| Group | Count | Examples |
|---|---|---|
| Lag features (1, 2, 3 months) | ~111 | `ICSA_lag1`, `UNRATE_lag2` |
| Rolling means (3M, 6M, 12M) | ~111 | `AWHAETP_ma12`, `CIVPART_ma6` |
| Z-scores (24M window) | ~37 | `TCU_zscore`, `CCSA_zscore` |
| Rolling std dev (6M) | ~37 | `INDPRO_std6` |
| Acceleration | ~37 | `UNRATE_accel` |
| Cross-series features | 13 | `yield_curve_10y2y`, `real_rate`, `adp_nfp_spread` |
| Calendar & regime | 11 | `is_recession`, `rate_hiking`, `vix_high_stress` |
| **Total** | **469** | |

**Output:** Bias-corrected point forecast with ±1σ confidence range saved to `data/forecast_history.csv`

---

### Stage 2 — Surprise Signal

**Script:** `surprise_model.py`  
**Models:** XGBoost Classifier (beat/miss) + XGBoost Regressor (surprise magnitude)

Answers the question markets actually care about: will this NFP beat or miss consensus?

**Key alpha signals:**
- `model_vs_consensus` — gap between Stage 1 forecast and market consensus
- `consensus_bias_3m/6m/12m` — systematic consensus under/over-estimation
- `beats_last_3m/6m` — recent beat streak momentum
- `surprise_lag1/2/3` — historical surprise autocorrelation

**Surprise buckets:**
| Bucket | Condition |
|---|---|
| LARGE MISS | Surprise < −50K |
| SMALL MISS | −50K ≤ Surprise < 0 |
| SMALL BEAT | 0 ≤ Surprise < +50K |
| LARGE BEAT | Surprise ≥ +50K |

**Output:** Beat probability (0–100%), predicted surprise magnitude, BEAT/MISS signal with HIGH/MEDIUM/LOW confidence

---

### Stage 3 — Market Reaction

**Script:** `market_reaction_model.py`

Maps the predicted surprise bucket to historical same-day market moves across 5 assets.

**Assets covered:**
| Asset | Ticker |
|---|---|
| Gold | GC=F |
| EUR/USD | EURUSD=X |
| US Dollar Index | DX-Y.NYB |
| 10-Year Treasury Yield | ^TNX |
| S&P 500 | ^GSPC |

**Output per asset:** Up% rate, average move, median move, 25th–75th percentile range — all conditioned on surprise bucket

---

## Dashboard

### Streamlit (interactive, live)

```bash
streamlit run dashboard.py
```

Features:
- Live consensus input with real-time signal recalculation
- Stage 2 beat probability gauge
- Stage 3 asset reaction cards
- Walk-forward accuracy by year
- Top 15 feature importances
- Stage 2 prediction history table

### HTML Report (static, shareable)

Generated automatically at `reports/nfp_forecast_report.html` when you run `run_forecast.py`. Open directly in any browser — no server required.

---

## Model Performance

**Stage 1 Walk-Forward (2011–2026, COVID excluded):**

| Period | MAE | Hit Rate |
|---|---|---|
| 2011–2019 | ~75–120K | 100% |
| 2020 (COVID) | ~1,078K | 42% (excluded from training) |
| 2021–2022 | ~290–468K | 91–100% |
| 2023–2024 | ~112–118K | 92–100% |
| 2025 | ~90K | 67% |

**Stage 2 Walk-Forward (2018–2026):**

| Metric | Value |
|---|---|
| Overall direction accuracy | 52.6% |
| 2026 accuracy | 5/5 correct (100%) |
| High-confidence signal accuracy | 52.1% |
| Surprise MAE | ±127K |

> COVID months (Mar–Aug 2020) are excluded from training data and performance metrics as they represent a structural break outside the model's intended regime.

---

## Data Sources

All macroeconomic data is fetched from the [FRED API](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis). Market reaction data (Stage 3) is downloaded from Yahoo Finance via `yfinance`.

**Key series included:**

| Category | Series |
|---|---|
| Labor market | PAYEMS, ICSA, CCSA, UNRATE, CIVPART, EMRATIO, JOLTS |
| Wages & hours | AWHAETP, CES0500000003 |
| Leading indicators | ADPMNUSNERSA, NFCI, UMCSENT, CSCICP03USM665S |
| Inflation | CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, PPIACO |
| Interest rates | DGS2, DGS10, FEDFUNDS, T10Y2Y |
| Financial conditions | VIXCLS, DTWEXBGS, M2SL |
| Real activity | INDPRO, TCU, HOUST, PERMIT, RSAFS, DGORDER |
| Regime indicator | USREC (NBER recession dates) |

---

## Configuration

All parameters are in `config.py`:

```python
START_DATE = "2000-01-01"       # Training data start
FRED_API_KEY = "..."            # Set via .env file

LAG_PERIODS = [1, 2, 3]        # Lag windows for feature engineering
ROLLING_WINDOWS = [3, 6, 12]   # Rolling mean windows (months)
ZSCORE_WINDOW = 24              # Z-score lookback window
```

**Walk-forward settings** (in `model_trainer.py`):
```python
WF_START_YEAR = 2011            # First out-of-sample prediction year
WINSOR_THRESHOLD = 1500         # Clip target beyond ±1500K (handles COVID)
```

---

## Dependencies

```
fredapi==0.5.2          # FRED API client
pandas>=2.1.0           # Data manipulation
numpy>=1.26.0           # Numerical computing
xgboost>=2.0.0          # Gradient boosted trees
scikit-learn>=1.4.0     # Preprocessing and metrics
shap>=0.45.0            # Feature importance (SHAP values)
yfinance                # Yahoo Finance (Stage 3 market data)
streamlit               # Interactive dashboard
plotly>=5.19.0          # Charts in Streamlit dashboard
pyarrow>=14.0.0         # Parquet file I/O
colorama>=0.4.6         # Colored terminal output
tqdm>=4.66.0            # Progress bars
python-dotenv           # Environment variable loading
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

> **Disclaimer:** This system is for research and educational purposes. Historical patterns do not guarantee future market reactions. NFP forecasts carry substantial uncertainty (±144K MAE under normal conditions). Do not use as the sole basis for trading decisions.
