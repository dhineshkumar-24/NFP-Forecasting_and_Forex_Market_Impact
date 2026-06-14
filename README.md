# Non-Farm Payrolls (NFP) Forecasting System

A Python-based system designed to collect macroeconomic indicators, engineer predictive features, train forecasting models, and generate live US Non-Farm Payrolls (NFP) nowcast predictions.

---

## Folder Structure

```text
nfp_forecaster/
├── config.py                       # API configs, series dictionary, and parameters
├── data_collector.py               # Step 1: Downloads and merges macroeconomic data
├── feature_engineer.py             # Step 2: Engineers predictive features & handles outliers
├── model_trainer.py                # Step 3: Trains model with walk-forward validation
├── nowcaster.py                    # Step 4: Generates live NFP nowcasts
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── data/
│   ├── raw/                        # Individual raw Parquet series files
│   ├── processed_macro_data.parquet # Consolidated wide month-end dataset
│   ├── feature_matrix.parquet      # Engineered features matrix for training
│   ├── walk_forward_results.csv    # Historical walk-forward predictions vs actuals
│   └── forecast_history.csv        # Live forecast run history
└── models/
    ├── nfp_model.pkl               # Trained XGBoost regressor model
    ├── imputer.pkl                 # SimpleImputer object
    └── model_metadata.json         # Model parameters, metrics, and feature importances
```

---

## Setup Instructions

### 1. Create a Virtual Environment
Navigate to the project directory and create a Python virtual environment:
```bash
python -m venv nfp_env
```

### 2. Activate the Virtual Environment
* **On Windows (Command Prompt):**
  ```cmd
  nfp_env\Scripts\activate.bat
  ```
* **On Windows (PowerShell):**
  ```powershell
  nfp_env\Scripts\activate.ps1
  ```
* **On macOS/Linux:**
  ```bash
  source nfp_env/bin/activate
  ```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure FRED API Key
Create a `.env` file in the root of the project directory and add your FRED API key:
```env
FRED_API_KEY=your_actual_api_key_here
```

---

## How to Run

### Step 1: Collect Macro Data
Downloads the 38 leading economic indicators from the FRED API, resamples them to month-end, and merges them:
```bash
python data_collector.py
```
*Note: To run offline or generate synthetic datasets for testing without a FRED key:*
```bash
python data_collector.py --mock
```

### Step 2: Feature Engineering
Generates 468 predictive features (lags, rolling statistics, cross-series spreads, calendar and recession regime metrics) and cleans infinite values:
```bash
python feature_engineer.py
```

### Step 3: Model Training
Trains the XGBoost models using expanding-window walk-forward validation, evaluates COVID-included and COVID-excluded strategies, and saves the winner:
```bash
python model_trainer.py
```

### Step 4: Live Nowcast
Generates the live forecast for the upcoming NFP release, incorporating bias correction and confidence intervals from walk-forward history:
```bash
python nowcaster.py
```

---

## Monthly Workflow

To keep forecasts accurate and up to date:
1. **Frequency:** Run the pipeline **once a month on Thursday** directly preceding **NFP Friday** (usually the first Friday of the month).
2. **Timing:** Run the pipeline **after the ADP National Employment Report is released** (usually Wednesday morning). This ensures all the latest weekly jobless claims, ADP payroll proxies, and daily index readings for the ending month are captured.
3. **Execution Sequence:**
   ```powershell
   python data_collector.py
   python feature_engineer.py
   python model_trainer.py
   python nowcaster.py
   ```
4. **Accuracy Tracking:** After the official NFP release on Friday morning, open `data/forecast_history.csv` and manually update the `actual_nfp` and `error` columns to track forecasting performance over time.
