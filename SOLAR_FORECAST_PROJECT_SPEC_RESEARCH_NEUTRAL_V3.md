# Solar Energy Forecasting — Full Project Specification
> **Purpose:** This document is a complete, self-contained specification for an AI coding agent to build the entire solar energy forecasting project from scratch. Every requirement, data fact, formula, threshold, file name, column name, metric, and expected output is stated explicitly. No assumptions should be made beyond what is written here.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Specification](#3-dataset-specification)
4. [Data Ingestion & Splitting](#4-data-ingestion--splitting)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Definitions](#8-model-definitions)
9. [Training Procedure](#9-training-procedure)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [24-Hour Rollout Forecast](#11-24-hour-rollout-forecast)
12. [Factor Dominance Analysis](#12-factor-dominance-analysis)
13. [Visualisation Requirements](#13-visualisation-requirements)
14. [Experimental Results](#14-experimental-results)
15. [Output Files](#15-output-files)
16. [Springer Paper Structure](#16-springer-paper-structure)
17. [Implementation Notes & Constraints](#17-implementation-notes--constraints)

---

## 1. Project Overview

### Goal
Predict `Energy delta[Wh]` (solar energy generated per 15-minute interval) for the **next 24 hours** (96 time steps) using multiple linear regression models trained on historical weather and energy data.

### Deliverables
1. A fully reproducible Python pipeline (`pipeline.py`)
2. Two split dataset files: `training.xlsx` and `testing.xlsx`
3. Nine visualisation figures (PNG, 150 dpi minimum)
4. A Springer-format conference paper (`paper.md` or `paper.docx`)

### Assignment constraints (non-negotiable)
- The final forecasting model should be selected after empirical comparison using the chosen evaluation metrics.
- Strict train/test separation: `training.xlsx` (2017–2021), `testing.xlsx` (2022)
- TModels must be built and compared
- Evaluation metrics: **MSE, MAE, RMSE, Relative Error** — all four required
- Forecast horizon: **24 hours at 15-minute intervals = 96 steps**
- Target relative error: ≤ 5%

---

## 2. Repository Structure

```
project/
├── data/
│   ├── Renewable.csv                    # raw input (provided)
│   ├── training.xlsx                    # split output: 2017-01-01 → 2021-12-31
│   ├── testing.xlsx                     # split output: 2022-01-01 → 2022-08-31
│   ├── training_filled_gaps.csv         # gap-filled timestamps (training)
│   └── testing_filled_gaps.csv          # gap-filled timestamps (testing)
├── outputs/
│   ├── 01_correlation.png
│   ├── 02_scatter_grid.png
│   ├── 03_model_comparison_table.png
│   ├── 04_error_comparison.png
│   ├── 05_forecast_vs_actual.png
│   ├── 06_residuals.png
│   ├── 07_feature_importance.png
│   ├── 08_2week_forecast.png
│   └── 09_full_results_dashboard.png
├── pipeline.py                # single-file reproducible pipeline
├── paper.md                   # Springer conference paper
├── SOLAR_FORECAST_PROJECT_SPEC_RESEARCH_NEUTRAL_V3.md   # this file
└── AGENTS.md                  # agent instructions / session log
```

---

## 3. Dataset Specification

### Source file
`data/Renewable.csv` — comma-separated, no BOM, Windows line endings (`\r\n`)

### Shape
- **196,776 rows × 17 columns**
- Time span: `2017-01-01 00:00` → `2022-08-31 17:45`
- Nominal frequency: 15-minute intervals
- **1,824 missing timestamps** (sensor outages) — must be handled in preprocessing

### Column definitions

| Column | Type | Unit | Description |
|---|---|---|---|
| `Time` | string → datetime | — | Timestamp, format `M/D/YYYY H:MM` |
| `Energy delta[Wh]` | int64 | Wh | **TARGET**: energy generated in this 15-min interval |
| `GHI` | float64 | W/m² | Global Horizontal Irradiance — primary solar driver |
| `temp` | float64 | °C | Air temperature |
| `pressure` | int64 | hPa | Atmospheric pressure |
| `humidity` | int64 | % | Relative humidity |
| `wind_speed` | float64 | m/s | Wind speed |
| `rain_1h` | float64 | mm | Rainfall in past hour |
| `snow_1h` | float64 | mm | Snowfall in past hour |
| `clouds_all` | int64 | % | Cloud cover percentage |
| `isSun` | int64 | binary | 1 = sunlight present, 0 = night/overcast |
| `sunlightTime` | int64 | minutes | Cumulative sunlight minutes so far today |
| `dayLength` | int64 | minutes | Total daylight minutes for this date |
| `SunlightTime/daylength` | float64 | ratio [0,1] | Proportion of daylight that has been sunny |
| `weather_type` | int64 | category 1–5 | Encoded weather category |
| `hour` | int64 | 0–23 | Hour of day (already extracted) |
| `month` | int64 | 1–12 | Month of year (already extracted) |

### Key distributional facts (verified from data)

```
Total rows:                 196,776
Missing values in columns:  0  (all columns fully populated)
Missing timestamps:         1,824  (gaps in time index, not NaN values)

Target — Energy delta[Wh]:
  min:    0 Wh
  mean:   573 Wh
  max:    5,020 Wh
  zeros:  51.3% of rows  (night + zero-irradiance periods — physically correct, NOT errors)

GHI:
  zeros:  48.3% of rows  (matches isSun=0 exactly — confirms night periods)
  max:    229.2 W/m²

Pearson correlation with Energy delta[Wh]:
  GHI                     +0.918   (dominant)
  isSun                   +0.527
  sunlightTime            +0.441
  SunlightTime/daylength  +0.403
  temp                    +0.386
  dayLength               +0.291
  pressure                +0.113
  wind_speed              +0.032
  month                   -0.061
  snow_1h                 -0.067
  rain_1h                 -0.067
  hour                    -0.086
  weather_type            -0.177
  clouds_all              -0.199
  humidity                -0.547
```

---

## 4. Data Ingestion & Splitting

### Split rule
```
training.xlsx : rows where year(Time) in {2017, 2018, 2019, 2020, 2021}
testing.xlsx  : rows where year(Time) == 2022
```

### Expected sizes after split (before gap-filling)
```
training.xlsx : 174,048 rows   (2017-01-01 → 2021-12-31)
testing.xlsx  :  22,728 rows   (2022-01-01 → 2022-08-31 17:45)
```

### Code
```python
import pandas as pd

df = pd.read_csv("data/Renewable.csv")
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)

train_raw = df[df["Time"].dt.year <= 2021].copy()
test_raw  = df[df["Time"].dt.year == 2022].copy()

train_raw.to_excel("data/training.xlsx", index=False)
test_raw.to_excel("data/testing.xlsx",   index=False)

print(f"Train: {len(train_raw):,} rows")   # → 174,048
print(f"Test:  {len(test_raw):,} rows")    # → 22,728
```

### Strict separation rule
> **CRITICAL:** After this split, every subsequent operation that involves computing statistics, fitting transformers, or fitting models **must use `training.xlsx` data only**. Statistics computed on training data are then applied (not re-computed) on testing data. This prevents data leakage.

---

## 5. Preprocessing Pipeline

All steps below must be applied in order. Steps 5.1–5.4 are applied independently to train and test; statistics are always fit on train only.

### 5.1 Datetime parsing and sorting
```python
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)
```



### 5.2 Reindex to Complete 15-Minute DatetimeIndex

This is the most critical preprocessing step. Missing timestamps represent sensor outages rather than confirmed zero-generation periods.

Directly filling missing target values with zeros would artificially create false nighttime behavior inside daytime intervals, distort temporal autocorrelation, and severely damage downstream lag features.

Instead of using simple interpolation or recursive seasonal propagation alone, missing target values are reconstructed using historical pattern-based gap reconstruction.

This approach searches historical periods with similar temporal behavior and uses those cyclic patterns to reconstruct long missing intervals while preserving realistic photovoltaic generation dynamics.

```python
import pandas as pd
import numpy as np

def reindex_fill(df_in, target_col="Energy delta[Wh]"):

    d = df_in.set_index("Time").sort_index()

    # Build complete 15-minute DatetimeIndex
    full_idx = pd.date_range(
        start=d.index.min(),
        end=d.index.max(),
        freq="15min"
    )

    # Reindex missing timestamps
    d = d.reindex(full_idx)

    # ---------------------------------------------------------
    # WEATHER FEATURE IMPUTATION
    # ---------------------------------------------------------

    weather_cols = [c for c in d.columns if c != target_col]

    d[weather_cols] = d[weather_cols].ffill().bfill()

    # ---------------------------------------------------------
    # HISTORICAL PATTERN-BASED RECONSTRUCTION
    # ---------------------------------------------------------
    #
    # 15-minute interval:
    # 24 hours = 96 timesteps
    #
    # Strategy:
    # 1. Detect continuous missing gaps
    # 2. Extract previous 24h context
    # 3. Search historical windows with similar patterns
    # 4. Copy corresponding future sequence
    # 5. Scale reconstructed sequence if necessary
    #
    # This preserves:
    # - daily solar periodicity
    # - sunrise/sunset structure
    # - temporal autocorrelation
    # - realistic irradiance variability
    # ---------------------------------------------------------

    seasonal_window = 96

    is_missing = d[target_col].isna()

    gap_groups = (
        is_missing.ne(is_missing.shift())
        .cumsum()
    )

    for gap_id in gap_groups[is_missing].unique():

        gap_idx = d.index[gap_groups == gap_id]

        gap_start = gap_idx[0]
        gap_len = len(gap_idx)

        # Previous 24h context
        context_start = gap_start - pd.Timedelta(minutes=15 * seasonal_window)

        if context_start not in d.index:
            continue

        context = d.loc[
            context_start : gap_start - pd.Timedelta(minutes=15),
            target_col
        ].values

        if np.isnan(context).any():
            continue

        best_score = -np.inf
        best_future = None

        # Search historical windows
        for i in range(seasonal_window, len(d) - gap_len - seasonal_window):

            hist_context = d[target_col].iloc[
                i - seasonal_window : i
            ].values

            hist_future = d[target_col].iloc[
                i : i + gap_len
            ].values

            if (
                np.isnan(hist_context).any() or
                np.isnan(hist_future).any()
            ):
                continue

            # Correlation similarity
            score = np.corrcoef(context, hist_context)[0, 1]

            if score > best_score:

                best_score = score
                best_future = hist_future.copy()

        # Reconstruct missing sequence
        if best_future is not None:

            scale = (
                np.mean(context) /
                (np.mean(best_future[:seasonal_window]) + 1e-6)
            )

            reconstructed = best_future * scale

            d.loc[gap_idx, target_col] = reconstructed

    # ---------------------------------------------------------
    # FINAL CONTINUITY RECONSTRUCTION
    # ---------------------------------------------------------

    d[target_col] = d[target_col].interpolate(method="time")

    d[target_col] = d[target_col].ffill().bfill()

    # Physical constraint
    d[target_col] = d[target_col].clip(lower=0)

    assert d.isnull().sum().sum() == 0,         "Null values remain after preprocessing"

    return d

train_pp, train_missing = reindex_fill(train_raw)
test_pp, test_missing = reindex_fill(test_raw)

# Save only the timestamps that were missing (gap-filled rows)
train_pp[train_missing].to_csv("data/training_filled_gaps.csv")
test_pp[test_missing].to_csv("data/testing_filled_gaps.csv")
```

### Rationale

Missing timestamps correspond to sensor outages rather than true zero-generation periods.

Because photovoltaic datasets naturally contain nighttime zeros, directly inserting zeros into outage intervals creates unrealistic artificial nighttime behavior during daytime periods. This severely distorts:

- temporal autocorrelation,
- daily generation cycles,
- lag-based features,
- and rolling temporal statistics.

The reconstruction strategy therefore uses historical cyclic similarity instead of naive interpolation.

For each missing interval, the algorithm:

1. extracts the previous 24-hour temporal context,
2. searches historical periods with similar generation behavior,
3. identifies the most correlated historical sequence,
4. and reconstructs the missing interval using the matched future pattern.

This approach preserves:

- daily photovoltaic periodicity,
- sunrise and sunset continuity,
- irradiance fluctuation structure,
- and realistic temporal dynamics.

Long gaps can therefore be reconstructed more realistically than with interpolation or simple seasonal averaging alone.

The final interpolation stage is applied only to small unresolved residual gaps in order to maintain short-term continuity.


### 5.3 Outlier clipping

Use **percentile-based clipping**: fit bounds on training data, apply same bounds to test.

```python
clip_config = {
    "GHI":        (0.01, 0.99),   # bounds: [0.00, 200.70]
    "temp":       (0.01, 0.99),   # bounds: [-7.10, 28.10]
    "wind_speed": (0.01, 0.99),   # bounds: [0.80, 8.90]
    "rain_1h":    (0.01, 0.99),   # bounds: [0.00, 1.34]
    "snow_1h":    (0.01, 0.99),   # bounds: [0.00, 0.28]
    "clouds_all": (0.01, 0.99),   # bounds: [0.00, 100.00]  (no change)
}

clip_bounds = {}
for col, (lo_q, hi_q) in clip_config.items():
    lo = train[col].quantile(lo_q)
    hi = train[col].quantile(hi_q)
    clip_bounds[col] = (lo, hi)
    train[col] = train[col].clip(lo, hi)

# Apply SAME bounds from train to test — never recompute on test
for col, (lo, hi) in clip_bounds.items():
    test[col] = test[col].clip(lo, hi)
```

**Clipped row counts (training):**
- GHI: 1,745 rows
- temp: 3,416 rows
- wind_speed: 3,196 rows
- rain_1h: 1,740 rows
- snow_1h: 1,688 rows
- clouds_all: 0 rows

### 5.4 Feature scaling

Use `StandardScaler` — fit on training, transform both sets. The **target column is NOT scaled** (keeps error metrics in original Wh units).

```python
from sklearn.preprocessing import StandardScaler

# Scale only these columns — NOT the target, NOT binary/categorical columns
scale_cols = ["GHI", "temp", "pressure", "humidity", "wind_speed",
              "rain_1h", "snow_1h", "clouds_all", "sunlightTime",
              "dayLength", "SunlightTime/daylength"]

scaler_input = StandardScaler()
train[scale_cols] = scaler_input.fit_transform(train[scale_cols])   # fit + transform
test[scale_cols]  = scaler_input.transform(test[scale_cols])        # transform only
```

> Note: The per-model `StandardScaler` in §9 replaces this step for the engineered feature matrix. This global scaler is for reference; the per-model scalers in §9 take priority.

---

## 6. Exploratory Data Analysis

EDA outputs are **not** used to inform model training (no leakage). They are computed on `train_fe` (training data after feature engineering) and saved as figures.

### 6.1 Pearson correlation matrix
- Compute `train_fe[numeric_cols].corr()`
- Plot as lower-triangular heatmap with `seaborn.heatmap`, `cmap="coolwarm"`, `center=0`, `annot=True`, `fmt=".2f"`, font size 7
- Also plot a horizontal bar chart of `corr["Energy delta[Wh]"]` sorted ascending, with negative bars in `#E05252` and positive in `#378ADD`
- Save as `outputs/01_correlation.png`

### 6.2 Scatter grid
- 8 subplots (2×4) of EnergyDelta vs each: `GHI, SunlightTime/daylength, sunlightTime, isSun, clouds_all, temp, humidity, wind_speed`
- `alpha=0.03, s=0.5, color="#378ADD"`
- Each subplot title: `r = {pearson_r:.3f}`
- Save as `outputs/02_scatter_grid.png`

### 6.3 EDA findings to discuss in paper

Discuss the observed relationships between variables based on the computed EDA results.
Report positive and negative correlations, distributional characteristics,
potential multicollinearity, and any relevant insights discovered from the data.

---

## 7. Feature Engineering

Apply the `engineer()` function to **both** train and test **after** preprocessing. This function only uses `shift()` and `rolling()` — it does not compute any statistics that need to be fit separately.

```python
import numpy as np

def engineer(df_in, target_col="Energy delta[Wh]"):
    d = df_in.copy()

    # --- Lag features (temporal memory) ---
    d["ED_lag1"]   = d[target_col].shift(1)    # EnergyDelta 15 min ago
    d["ED_lag4"]   = d[target_col].shift(4)    # EnergyDelta 1 hour ago
    d["ED_lag96"]  = d[target_col].shift(96)   # EnergyDelta same time yesterday
    d["GHI_lag1"]  = d["GHI"].shift(1)         # GHI 15 min ago

    # --- Rolling statistics ---
    d["GHI_roll4"] = d["GHI"].rolling(window=4, min_periods=1).mean()        # 1-hour rolling GHI mean
    d["ED_roll4"]  = d[target_col].rolling(window=4, min_periods=1).mean()   # 1-hour rolling energy mean

    # --- Cyclic time encoding (avoids 23→0 discontinuity) ---
    d["hour_sin"]  = np.sin(2 * np.pi * d["hour"]  / 24)
    d["hour_cos"]  = np.cos(2 * np.pi * d["hour"]  / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    # --- Interaction terms ---
    d["GHI_x_sun"]   = d["GHI"] * d["SunlightTime/daylength"]   # GHI weighted by sunlight fraction
    d["GHI_x_isSun"] = d["GHI"] * d["isSun"]                    # GHI masked to sunlight hours

    # Drop NaN rows created by shift(96) — affects first 96 rows (~1 day)
    return d.dropna()
```

**Post-engineering sizes:**
- `train_fe`: 175,200 rows
- `test_fe`: 23,208 rows

> The `dropna()` removes approximately 96 rows from the start. **Do not impute lags** — this would introduce look-ahead bias.

---



## 8. Model Definitions

This project uses a single forecasting model selected based on preliminary benchmarking experiments using Relative Error (%), RMSE, and generalization performance on unseen solar energy data.

Compare candidate forecasting models and select the best-performing approach based on experimental results.

The model is designed to learn:

- nonlinear weather-energy relationships,
- temporal autocorrelation,
- cyclic solar generation behavior,
- and short-term forecasting dynamics.

All preprocessing and feature engineering stages are optimized specifically for this final model architecture.

---

### 8.1 Final Feature Set

```python
FEATURES = [

    # Weather features
    "GHI",
    "temp",
    "pressure",
    "humidity",
    "wind_speed",
    "rain_1h",
    "snow_1h",
    "clouds_all",

    # Solar state
    "isSun",
    "sunlightTime",
    "dayLength",
    "SunlightTime/daylength",

    # Cyclic time encoding
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # Temporal lag features
    "ED_lag1",
    "ED_lag4",
    "ED_lag96",

    # Irradiance temporal memory
    "GHI_lag1",
    "GHI_roll4",

    # Energy rolling statistics
    "ED_roll4",

    # Interaction terms
    "GHI_x_sun",
    "GHI_x_isSun"
]
```

### Feature Engineering Rationale

The feature set combines:

- meteorological conditions,
- cyclic temporal encoding,
- autoregressive lag memory,
- rolling temporal statistics,
- and nonlinear interaction effects.

These features are necessary because photovoltaic generation exhibits strong daily periodicity, temporal continuity, and nonlinear responses to weather variability.

---

### 8.2 Final Forecasting Model — LightGBM

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Model Characteristics

- Gradient boosting decision tree model
- Strong nonlinear learning capability
- Handles complex feature interactions automatically
- Efficient for large tabular time-series datasets
- Robust against noisy weather fluctuations
- Well-suited for short-term renewable energy forecasting

---

### 8.3 Training Procedure

```python
TARGET = "Energy delta[Wh]"

X_train = train_fe[FEATURES]
y_train = train_fe[TARGET]

X_test  = test_fe[FEATURES]
y_test  = test_fe[TARGET]
```

Feature scaling is not required because LightGBM is tree-based and insensitive to feature magnitude.

---

### 8.4 Evaluation Metrics

Model performance is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Relative Error (%)
- Coefficient of Determination (R²)

Primary evaluation criterion:

$$
RelError(\%) = rac{RMSE}{mean(y_{true})} 	imes 100
$$

Relative Error is prioritized because it measures forecasting deviation relative to the average solar generation magnitude.

---

### 8.5 Final Forecasting Workflow

The final workflow is:

1. Data preprocessing and gap reconstruction
2. Temporal feature engineering
3. LightGBM model training
4. Multi-step recursive forecasting
5. Residual analysis and evaluation
6. 24-hour solar energy prediction


## 9. Training Procedure

For each model, apply this identical procedure:

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

TARGET = "Energy delta[Wh]"

def train_model(train_fe, feature_list):
    X_train = train_fe[feature_list].values
    y_train = train_fe[TARGET].values

    # Fit scaler on training features only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit OLS model
    model = LinearRegression()
    model.fit(X_scaled, y_train)

    train_r2 = model.score(X_scaled, y_train)
    print(f"Training R²: {train_r2:.4f}")

    return model, scaler

model_A, scaler_A = train_model(train_fe, FEATS_A)
model_B, scaler_B = train_model(train_fe, FEATS_B)
model_C, scaler_C = train_model(train_fe, FEATS_C)
```



> Predictions must be **clipped to ≥ 0**: `y_pred = model.predict(X_scaled).clip(0)`. Energy generation cannot be negative; OLS occasionally predicts small negative values near dawn/dusk.

---

## 10. Evaluation Metrics

All four metrics must be computed and reported for both training and test sets, for all three models.

### Definitions

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_true, y_pred, label=""):
    y_pred = y_pred.clip(0)   # enforce non-negativity

    mse   = mean_squared_error(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    rel   = (rmse / y_true.mean()) * 100        # Relative Error (%)
    r2    = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

    print(f"{label}")
    print(f"  MSE  = {mse:>12.1f} Wh²")
    print(f"  MAE  = {mae:>12.1f} Wh")
    print(f"  RMSE = {rmse:>12.1f} Wh")
    print(f"  RelErr = {rel:>8.2f} %    (target: ≤ 5%)")
    print(f"  R²   = {r2:>12.4f}")
    return dict(mse=mse, mae=mae, rmse=rmse, rel=rel, r2=r2)
```

### Relative Error formula
```
RelError (%) = (RMSE / mean(y_true)) × 100
```
where `mean(y_true)` is computed over **all timesteps** in the evaluated set (including night zeros).

### Secondary metric: daytime-only RelError
Because 47.4% of test values are zero (night periods), the standard RelError appears inflated relative to solar forecasting industry norms. Compute and report a secondary metric:

```python
def compute_daytime_rel_error(y_true, y_pred, ghi_values):
    """Evaluates only on timesteps where GHI > 0 (sunlight present)."""
    daytime_mask = ghi_values > 0
    y_d  = y_true[daytime_mask]
    yp_d = y_pred[daytime_mask].clip(0)
    rmse_d = np.sqrt(mean_squared_error(y_d, yp_d))
    return (rmse_d / y_d.mean()) * 100
```



> In the paper, discuss both metrics. Use the standard RelError for the formal comparison table. Note in the discussion that the daytime-only metric is more appropriate for solar generation assessment.

---

## 11. 24-Hour Rollout Forecast

This implements a **multi-step autoregressive rollout**: predicted values are fed back as lag features for subsequent steps.

### Forecast setup
- **96 time steps × 15 minutes = 24 hours**
- Weather features (`GHI`, `temp`, etc.) for the forecast period are taken directly from `test_fe` (treat as known forecast inputs — analogous to a NWP weather forecast being available)
- Lag features are initialised from the last known historical values, then updated with predictions at each step

### Algorithm

```python
def forecast_24h(model, scaler, feature_list, test_df, history_df,
                 start_pos=0, target_col="Energy delta[Wh]"):
    """
    Parameters
    ----------
    model       : fitted LinearRegression
    scaler      : fitted StandardScaler (same as used in training)
    feature_list: list of feature names for this model
    test_df     : DataFrame of test features (engineered, indexed by datetime)
    history_df  : DataFrame of training features — used to seed lag buffers
    start_pos   : integer row index in test_df to start the 24-hour forecast from
    """

    # Initialise rolling buffers from history (need last 96 steps for ED_lag96)
    buf_ed  = list(history_df[target_col].values[-200:])   # energy buffer
    buf_ghi = list(history_df["GHI"].values[-200:])        # GHI buffer

    predictions = []

    for step in range(96):
        row = test_df.iloc[start_pos + step].copy()

        # Override lag features with buffer values (autoregressive)
        if "ED_lag1"  in feature_list: row["ED_lag1"]  = buf_ed[-1]
        if "ED_lag4"  in feature_list: row["ED_lag4"]  = buf_ed[-4]
        if "ED_lag96" in feature_list: row["ED_lag96"] = buf_ed[-96] if len(buf_ed) >= 96 else 0.0
        if "GHI_lag1" in feature_list: row["GHI_lag1"] = buf_ghi[-1]
        if "GHI_roll4" in feature_list: row["GHI_roll4"] = float(np.mean(buf_ghi[-4:]))
        if "ED_roll4"  in feature_list: row["ED_roll4"]  = float(np.mean(buf_ed[-4:]))

        # Recompute interaction terms using current-step weather (from test_df)
        if "GHI_x_sun"   in feature_list: row["GHI_x_sun"]   = row["GHI"] * row["SunlightTime/daylength"]
        if "GHI_x_isSun" in feature_list: row["GHI_x_isSun"] = row["GHI"] * row["isSun"]

        # Predict
        X = np.array([[row[f] for f in feature_list]])
        y_pred = float(max(0.0, model.predict(scaler.transform(X))[0]))

        predictions.append(y_pred)

        # Update buffers with predicted value (autoregressive)
        buf_ed.append(y_pred)
        buf_ghi.append(float(row["GHI"]))

    return np.array(predictions)
```

### Forecast day selection
Select a representative forecast day using a clearly documented and reproducible criterion:

```python
test_daily_sum = test_fe[TARGET].resample("D").sum()
best_day = test_daily_sum.idxmax()   # 
```

### 
> **Important note for paper:** The rollout introduces cumulative error propagation because prediction errors in lag features compound over 96 steps. This is a known limitation of autoregressive linear models and should be stated explicitly in the Results & Discussion section.

---

## 12. Factor Dominance Analysis

### Standardised β coefficients

Standardised β allows comparison of feature importance across features with different scales:

```
β_std_i = β_i × (σ_xi / σ_y)
```

where `β_i` is the OLS coefficient, `σ_xi` is the standard deviation of feature `i` in the training set, and `σ_y` is the standard deviation of the target in the training set.

```python
def compute_importance(model, scaler, feature_list, train_fe, target_col="Energy delta[Wh]"):
    std_y = train_fe[target_col].std()
    std_X = train_fe[feature_list].std()
    betas_standardised = np.abs(model.coef_ * (std_X.values / std_y))
    return pd.Series(betas_standardised, index=feature_list).sort_values(ascending=False)
```

### Feature Importance Analysis

Compute and report feature importance rankings from the trained model. Do not assume any predefined ranking, dominant feature, or coefficient magnitude. All rankings and interpretations must be derived from experimental results.

### VIF analysis (multicollinearity check)

Compute Variance Inflation Factor for Model C features:

```python
from sklearn.linear_model import LinearRegression as LR

def compute_vif(train_fe, feature_list):
    X = train_fe[feature_list].values
    vifs = []
    for i, feat in enumerate(feature_list):
        X_others = np.delete(X, i, axis=1)
        r2 = LR().fit(X_others, X[:, i]).score(X_others, X[:, i])
        vif = 1 / (1 - r2) if r2 < 1 else 999
        vifs.append((feat, vif))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
```

**VIF Analysis Results**

Report VIF values computed from the experimental feature set.
Identify any multicollinearity issues based on observed results.
> **Paper discussion:** High VIF does not invalidate predictions (OLS predictions are still BLUE under Gauss-Markov) but it inflates coefficient standard errors. For a conference paper, acknowledge this and state that Model B addresses it through feature reduction.

---

## 13. Visualisation Requirements

All figures saved to `outputs/` at ≥ 150 dpi. Colour palette:
```
Model A: #7F77DD  (purple)
Model B: #EF9F27  (amber)
Model C: #1D9E75  (teal)
Reference lines (5% target, zero): red (#E05252)
```

### Figure 01 — Correlation analysis (`01_correlation.png`)
- **Left subplot:** Lower-triangular Pearson heatmap, `seaborn.heatmap`, annotated, coolwarm colormap
- **Right subplot:** Horizontal bar chart of `corr[TARGET]` sorted ascending; negative bars `#E05252`, positive `#378ADD`
- Size: 18×7 inches

### Figure 02 — Scatter grid (`02_scatter_grid.png`)
- 2×4 subplots; features: GHI, SunlightTime/daylength, sunlightTime, isSun, clouds_all, temp, humidity, wind_speed vs EnergyDelta
- `alpha=0.03, s=0.5`; title each subplot with `r = {pearson_r:.3f}`
- Size: 16×8 inches

### Figure 03 — Model comparison table (`03_model_comparison_table.png`)
- `matplotlib` table with columns: Model, Features, Train RMSE, Train RelErr, Test MSE, Test MAE, Test RMSE, Test RelErr, Test R²
- Header row: dark purple `#3C3489`, white text
- Model C row: light green background (best model)
- Size: 13×3.5 inches

### Figure 04 — Error metric bar comparison (`04_error_comparison.png`)
- 3 side-by-side bar charts: RMSE, MAE, RelError
- Each shows train (semi-transparent) and test (solid) for Models A, B, C
- RelError chart includes red dashed line at 5%
- Size: 14×5 inches

### Figure 05 — 24-hour forecast vs actual (`05_forecast_vs_actual.png`)
- 3 stacked subplots (one per model), shared x-axis
- Black solid line = actual; coloured dashed = predicted; shaded fill between them
- X-axis: 96 timestamps at 15-min spacing on best forecast day (2022-05-18)
- Each subplot title includes RelErr
- Size: 13×10 inches

### Figure 06 — Residual analysis (`06_residuals.png`)
- 3 subplots for Model C test predictions:
  1. Scatter: predicted vs residual (y=0 reference line)
  2. Histogram of residuals (80 bins)
  3. Q-Q plot against normal distribution
- Size: 15×4 inches

### Figure 07 — Feature importance (`07_feature_importance.png`)
- 3 horizontal bar charts (one per model), top-12 features by |β_std|
- Features with |β_std| > dominant threshold highlighted in model colour; others in `#CCCCCC`
- Size: 17×5 inches

### Figure 08 — 2-week time series (`08_2week_forecast.png`)
- 3 stacked subplots (one per model)
- Shows first 14 days of test set (96×14 = 1,344 steps)
- Actual (black, lw=0.5) + predicted (model colour, lw=0.6, dashed)
- Size: 15×10 inches

### Figure 09 — Full results dashboard (`09_full_results_dashboard.png`)
- 4×2 grid layout using `matplotlib.gridspec.GridSpec`
- Contains: RMSE bars, RelErr bars, 24-h forecast (all models), 2-week series, Model C importance, Model C residual histogram
- Size: 18×22 inches
- Suptitle: `"Solar Energy Prediction — Full Pipeline Results\nTraining: 2017–2021  |  Testing: Jan–Aug 2022  |  3 Linear Regression Models"`

---

## 14. Experimental Results

### 14.1 Baseline Comparison (Top-5 Features)

| Model | Type | Test RMSE | Test RelErr | Test R² | Daytime RelErr |
|---|---|---|---|---|---|
| Linear Regression | linear | 480.4 Wh | 69.55% | 0.8176 | 51.14% |
| Decision Tree | tree | 410.1 Wh | 59.37% | 0.8671 | 43.28% |
| Random Forest | forest | 384.2 Wh | 55.63% | 0.8833 | 40.37% |
| LightGBM | lightgbm | 377.9 Wh | 54.71% | 0.8872 | 39.67% |
| XGBoost | xgboost | 384.2 Wh | 55.62% | 0.8833 | 40.38% |
| MLP | mlp | **375.1 Wh** | **54.31%** | **0.8888** | **39.37%** |
| LightGBM (tuned) | lightgbm_tuned | 384.8 Wh | 55.71% | 0.8830 | 40.45% |
| MLP (tuned) | mlp_tuned | 375.9 Wh | 54.42% | 0.8883 | 39.46% |

### 14.2 Feature Expansion Experiment (LightGBM, 24 features)

| Metric | Top-5 Baseline | Expanded (24 feats) | Change |
|---|---|---|---|
| Test RMSE | 377.9 Wh | **210.8 Wh** | −44% |
| Test RelErr | 54.71% | **30.52%** | −24 pts |
| Test R² | 0.8872 | **0.9649** | +0.078 |
| Daytime RelErr | 39.67% | **22.76%** | −17 pts |
| 24h Forecast RelErr | 11.12% | 59.73% | +48.61 pts |

### 14.3 Daytime-Only Model (GHI > 0, Tuned LightGBM)

| Metric | Best Full Model | Daytime-Only | Improvement |
|---|---|---|---|
| RelErr (daytime-only) | 39.37% | **22.73%** | +16.64 pts |
| RMSE | 375.1 Wh | 282.6 Wh |
| R² | 0.8888 | **0.9503** |

### 14.4 Multi-Threshold Daytime Test

| GHI Threshold | Test Rows | Mean Target | RMSE | RelErr | R² |
|---|---|---|---|---|---|
| > 0 | 12,844 | 1,235 Wh | 282.6 Wh | 22.73% | 0.950 |
| > 25 | 8,927 | 1,703 Wh | 335 Wh | 19.67% | 0.943 |
| > 50 | 6,998 | 2,024 Wh | 365 Wh | 18.05% | 0.925 |
| > 100 | 4,141 | 2,592 Wh | 408 Wh | 15.75% | 0.884 |

As the GHI threshold rises, mean target increases (helping RelErr) but RMSE rises faster because fewer training examples remain and high-production hours have higher variance. The best RelErr (15.75% at GHI > 100) still exceeds 5%.

### 14.5 24-Hour Rollout Forecast (Top-5 Features, Best Day: 2022-05-18)

| Model | 24h RMSE | 24h RelErr |
|---|---|---|
| Linear Regression | 350.7 Wh | 27.38% |
| Decision Tree | 184.8 Wh | 14.42% |
| Random Forest | **134.5 Wh** | **10.49%** |
| LightGBM | 142.5 Wh | 11.12% |
| XGBoost | 176.9 Wh | 13.80% |
| MLP | 172.8 Wh | 13.49% |

### 14.6 Gap Reconstruction Statistics

| Dataset | Original Rows | After Reindex | Gap Rows Filled |
|---|---|---|---|
| Training | 174,048 | 175,296 | 1,248 |
| Testing | 22,728 | 23,304 | 576 |

### 14.7 Why 5% RelErr Is Unreachable

- **51.3% of all rows are zeros** (nighttime) — inflates denominator of `RMSE / mean(y_true) × 100`
- **58.9% of values < 100 Wh** — small denominators amplify errors at dawn/dusk
- Daytime-only filtering (GHI > 0) reduces RelErr to 22.73% but the model still has irreducible prediction error at 15-min resolution due to cloud-cover transitions
- Stricter GHI thresholds (> 50, > 100) improve RelErr to 15.75% but reduce available data by 75–87%, degrading model accuracy
- State-of-the-art 15-min solar forecasting achieves 20–40% RelErr; 5% would require near-perfect 15-min cloud prediction, which is physically impossible

## 15. Output Files

### Python pipeline (`pipeline.py`)
Single executable file. Running `python pipeline.py` from the project root must:
1. Read `data/Renewable.csv`
2. Create `data/training.xlsx` and `data/testing.xlsx`
3. Run all preprocessing, EDA, feature engineering, training, evaluation, forecasting, dominance analysis
4. Save gap-filled timestamps to `data/training_filled_gaps.csv` and `data/testing_filled_gaps.csv`
5. Save all 9 figures to `outputs/`
6. Print a final results summary table to stdout

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`, `lightgbm`, `xgboost`

Install command: `pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl lightgbm xgboost`

### Split data files
- `data/training.xlsx`: 174,048 rows × 17 columns (raw, pre-preprocessing)
- `data/testing.xlsx`: 22,728 rows × 17 columns (raw, pre-preprocessing)

### Gap-filled timestamp files
- `data/training_filled_gaps.csv`: ~1,248 rows — only the timestamps that were missing from the training index and reconstructed by the pattern-based gap-filling algorithm
- `data/testing_filled_gaps.csv`: ~576 rows — missing timestamps reconstructed for the test set

> Preprocessing is always applied in-memory from raw files; the split xlsx files store raw data only. Gap CSV files capture which rows were added during reindexing and what values were filled.

---

## 16. Springer Paper Structure

The paper must follow this structure exactly. All result values must match §14.

### Abstract (≤ 250 words)
Summarise: renewable energy dataset (196,776 rows, 2017–2022, 15-min interval); three OLS models; best model R² = <COMPUTED_R2>; dominant features are temporal lag terms and GHI; limitations of ≤5% target with linear regression.

### 1. Introduction
1. **Background and general problem:** Global energy transition, need for accurate renewable energy forecasting, role of solar power in smart grids
2. **Research field overview:** Short-term solar forecasting methods — physical, statistical, ML-based approaches; role of linear regression as interpretable baseline
3. **Literature review:** Previous studies using linear regression for solar forecasting; what has been achieved; open problems (zero-inflation, multi-step horizon, feature selection)
4. **Research objectives:**
   - Build three linear regression models with increasing feature complexity
   - Identify dominant weather and temporal predictors of EnergyDelta
   - Evaluate with MSE, MAE, RMSE, Relative Error on held-out 2022 data
   - Generate 24-hour, 96-step rollout forecasts
5. **Paper structure:** Brief 1-sentence summary of each remaining section

### 2. Materials and Methods

#### 2.1 Case Study — Dataset Description
- Source: Renewable Power Generation and Weather Conditions dataset
- 196,776 rows, 15-minute intervals, 2017-01-01 to 2022-08-31
- 17 columns (list all with units, from §3)
- Key characteristics: 51.3% zero target values, GHI as dominant driver (r = <COMPUTED_CORRELATION>), 1,824 missing timestamps handled by reindexing
- Train/test split: 2017–2021 

#### 2.2 Research Framework
- Include the 10-step workflow diagram (reference `09_full_results_dashboard.png`)
- Steps: Load → Reindex → Clip outliers → EDA → Feature engineering → Build models → Train evaluation → 24-h forecast → Test evaluation → Factor analysis

#### 2.3 Research Contents

**2.3.1 Theoretical basis — OLS Linear Regression**

Present the OLS equation:
```
ŷ = Xβ̂ + ε

β̂ = (XᵀX)⁻¹Xᵀy
```

State the Gauss-Markov assumptions: linearity, strict exogeneity, no perfect multicollinearity, homoscedasticity, no autocorrelation in residuals.

Acknowledge that assumption 5 (no autocorrelation) is violated by the time-series nature of solar data — this is the motivation for including lag features in Model C, which partially addresses the violation.

**2.3.2 Feature selection rationale**
- Model A: full feature set (baseline)
- Model B: correlation-based selection (top predictors, removes noise variables with r ≈ 0)
- Model C: adds autoregressive terms (ED_lag1, ED_lag4, ED_lag96) and interaction terms

**2.3.3 Evaluation metrics**
State formulas for all four: MSE, MAE, RMSE, Relative Error (exact formulas from §10).

**2.3.4 24-hour rollout methodology**
Explain the autoregressive rollout (§11), including the lag buffer strategy and the limitation of error propagation.

### 3. Results and Discussion

**3.1 EDA findings** — present and discuss the correlation analysis and observed variable relationships.

**3.2 Model training results** — Table with all train-set metrics (from §14)

**3.3 Model testing results** — Table with all test-set metrics; highlight Model C as best (R² = <COMPUTED_R2>, RMSE = 225.1 Wh)

**3.4 24-hour forecast analysis** — Figure 05; discuss rollout accuracy on best generation day (2022-05-18)

**3.5 Factor dominance** — Report experimentally derived feature importance results and discuss the observed ranking, dominant predictors, and their practical interpretation.

**3.6 Discussion of 5% RelError target** — explain zero-inflation issue; present daytime-only metric; contextualise R² = <COMPUTED_R2> as strong fit; Model C reduces RelError by 52% compared to Model A (67.92% → 32.93%)

### 4. Conclusion

1. Summary: three OLS models built, Model C (with lag features) best with R² = <COMPUTED_R2>, RMSE = 225.1 Wh
2. Key finding: temporal autocorrelation (lag features) contributes more to prediction accuracy than additional weather variables
3. GHI remains the dominant weather predictor; humidity the dominant negative predictor
4. Strength: interpretable linear model; computationally lightweight; transferable to other solar sites
5. Weakness: linear assumption violated by zero-inflation; cannot model abrupt weather transitions
6. Limitation: RelError of 32.93% exceeds 5% target due to zero-inflation in denominator
7. Future work: (a) separate day/night models, (b) LSTM for temporal dynamics, (c) quantile regression for uncertainty intervals, (d) real-time GHI forecasts as inputs

---

## 17. Implementation Notes & Constraints

### Language and libraries
- Python 3.8+
- `pandas >= 1.3`, `numpy >= 1.21`, `scikit-learn >= 0.24`, `matplotlib >= 3.4`, `seaborn >= 0.11`, `scipy >= 1.7`, `openpyxl >= 3.0`

### Reproducibility
Set random seeds where applicable (not needed for OLS but good practice):
```python
import numpy as np, random
np.random.seed(42); random.seed(42)
```

### Data leakage prevention — checklist
- [ ] `StandardScaler` fit only on `train_fe`, not `test_fe`
- [ ] Outlier clip bounds computed only from training quantiles
- [ ] `engineer()` uses only `shift()` and `rolling()` — no cross-sample statistics
- [ ] No rows from 2022 appear in training data at any point

### Prediction clipping
Always clip predictions to ≥ 0 before computing metrics:
```python
y_pred = model.predict(X_scaled).clip(0)
```

### DataFrame index
After `reindex_fill()`, the DataFrame index is a `DatetimeIndex` with freq='15min'. Use `.reset_index(drop=True)` only when integer indexing is needed for the rollout; otherwise preserve the DatetimeIndex for `.resample()` and time-based slicing.

### Memory management
The full training set after feature engineering is 175,200 rows × ~25 columns ≈ 35 MB. This fits comfortably in memory. No chunking or memory optimisation is needed.

### File paths
All paths should be relative. The pipeline assumes it is run from the `project/` root directory:
```python
import os
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
```

### Figures
- Use `matplotlib.use("Agg")` at the top of the script (non-interactive backend for headless environments)
- Always call `plt.close()` after `plt.savefig()` to prevent memory leaks
- Use `bbox_inches="tight"` in all `savefig()` calls

### Springer paper format requirements
- Font: Times New Roman or Computer Modern (LaTeX default)
- Two-column layout
- All figures referenced in text before they appear
- Tables formatted with `booktabs` (LaTeX) or equivalent
- Citations in numbered style [1], [2], ...
- Page limit: typically 10–12 pages for Springer LNCS format

---

*End of specification. All values, formulas, column names, file names, and expected outputs are stated explicitly above. An AI agent following this specification exactly will reproduce the pipeline and paper without ambiguity.*
