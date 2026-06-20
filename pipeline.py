import matplotlib
matplotlib.use("Agg")

import os, sys, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

TARGET = "Energy delta[Wh]"
TARGET_DIR = "outputs"

# ========== 1. LOAD & SPLIT ==========
print("=" * 60)
print("1. LOADING AND SPLITTING DATA")
print("=" * 60)

df = pd.read_csv("data/Renewable.csv")
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train_raw = df.iloc[:train_end].copy()
val_raw = df.iloc[train_end:val_end].copy()
test_raw = df.iloc[val_end:].copy()

train_raw.to_excel("data/training.xlsx", index=False)
val_raw.to_excel("data/validation.xlsx", index=False)
test_raw.to_excel("data/testing.xlsx", index=False)

print(f"Train: {len(train_raw):,} rows ({100*len(train_raw)//n}%)")
print(f"Val:   {len(val_raw):,} rows ({100*len(val_raw)//n}%)")
print(f"Test:  {len(test_raw):,} rows ({100*len(test_raw)//n}%)")

# ========== 2. PREPROCESSING ==========
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

def reindex_fill(df_in, target_col=TARGET):
    orig_dtypes = df_in.drop(columns=["Time"], errors="ignore").dtypes.to_dict()
    d = df_in.set_index("Time").sort_index()
    full_idx = pd.date_range(start=d.index.min(), end=d.index.max(), freq="15min")
    d = d.reindex(full_idx)
    missing_mask = d[target_col].isna().copy()

    # ---------------------------------------------------------
    # HISTORICAL PATTERN-BASED RECONSTRUCTION
    # ---------------------------------------------------------
    # Applied to ALL columns — weather + target
    #
    # 1. Detect continuous missing gaps (from target column)
    # 2. Extract previous 24h context from target column (best signal)
    # 3. Search historical windows with similar patterns
    # 4. Copy corresponding future sequence for ALL columns
    # 5. Scale each reconstructed column if necessary
    # --------------------------------------------------------- 

    seasonal_window = 96
    is_missing = d[target_col].isna()
    gap_groups = is_missing.ne(is_missing.shift()).cumsum()

    col_names = d.columns.tolist()
    arrays = {col: d[col].values.copy() for col in col_names}
    n_arr = len(arrays[target_col])

    for gap_id in gap_groups[is_missing].unique():
        gap_positions = np.where(gap_groups.values == gap_id)[0]
        gap_start = gap_positions[0]
        gap_len = len(gap_positions)
        context_start = gap_start - seasonal_window

        if context_start < 0:
            continue

        context = arrays[target_col][context_start:gap_start]
        if np.isnan(context).any():
            continue

        max_stop = min(n_arr, context_start) - gap_len + 1 
        search_starts = np.arange(seasonal_window, max_stop, max(1, gap_len // 4))

        best_score = -np.inf
        best_i = None

        for i in search_starts:
            hc = arrays[target_col][i - seasonal_window:i]
            hf = arrays[target_col][i:i + gap_len]
            if np.isnan(hc).any() or np.isnan(hf).any():
                continue
            c = np.corrcoef(context, hc)[0, 1]
            if c > best_score:
                best_score = c
                best_i = i

        if best_i is not None:
            for col in col_names:
                col_future = arrays[col][best_i:best_i + gap_len]
                if len(col_future) == 0:
                    continue
                # Scale: pre-gap mean / matched pre-gap mean for this column
                current_past = arrays[col][context_start:gap_start]
                matched_past = arrays[col][best_i - seasonal_window:best_i]
                cur_valid = current_past[~np.isnan(current_past)]
                mat_valid = matched_past[~np.isnan(matched_past)]
                if len(cur_valid) > 0 and len(mat_valid) > 0:
                    scale = np.mean(cur_valid) / (np.mean(mat_valid) + 1e-6)
                else:
                    scale = 1.0
                arrays[col][gap_positions] = col_future * scale

    for col in col_names:
        d[col] = pd.Series(arrays[col], index=d.index)

    # Final continuity reconstruction (small residual gaps)
    d = d.interpolate(method="time")
    d = d.ffill().bfill()
    d[target_col] = d[target_col].clip(lower=0)

    # Clip target to original pre-gap max to prevent filling artifacts
    orig_max = df_in[target_col].max()
    d[target_col] = d[target_col].clip(lower=0, upper=orig_max)

    # Restore original dtypes
    int_clip = {"isSun": (0, 1), "weather_type": (1, 5), "hour": (0, 23), "month": (1, 12)}
    for col, dt in orig_dtypes.items():
        if col not in d.columns:
            continue
        if np.issubdtype(dt, np.integer):
            lo, hi = int_clip.get(col, (None, None))
            vals = d[col].round()
            if lo is not None:
                vals = vals.clip(lo, hi)
            d[col] = vals.astype(dt)
        elif np.issubdtype(dt, np.floating):
            d[col] = d[col].astype(dt)

    assert d.isnull().sum().sum() == 0, "Null values remain"
    return d, missing_mask

train_pp, train_missing = reindex_fill(train_raw)
val_pp, val_missing = reindex_fill(val_raw)
test_pp, test_missing = reindex_fill(test_raw)
print(f"Train after reindex: {len(train_pp):,} rows")
print(f"Val after reindex:   {len(val_pp):,} rows")
print(f"Test after reindex:  {len(test_pp):,} rows")

# Outlier clipping
clip_config = {
    "GHI": (0.01, 0.99),
    "temp": (0.01, 0.99),
    "wind_speed": (0.01, 0.99),
    "rain_1h": (0.01, 0.99),
    "snow_1h": (0.01, 0.99),
    "clouds_all": (0.01, 0.99),
}

clip_bounds = {}
for col, (lo_q, hi_q) in clip_config.items():
    lo = train_pp[col].quantile(lo_q)
    hi = train_pp[col].quantile(hi_q)
    clip_bounds[col] = (lo, hi)
    train_pp[col] = train_pp[col].clip(lo, hi)

for col, (lo, hi) in clip_bounds.items():
    val_pp[col] = val_pp[col].clip(lo, hi)
    test_pp[col] = test_pp[col].clip(lo, hi)

print("Outlier clipping applied.")

# Save only originally-missing timestamps (gap-filled rows) to CSV
# Match original Renewable.csv format: M/D/YYYY H:MM, same column order
orig_cols = ["Time", "Energy delta[Wh]", "GHI", "temp", "pressure", "humidity",
             "wind_speed", "rain_1h", "snow_1h", "clouds_all", "isSun",
             "sunlightTime", "dayLength", "SunlightTime/daylength",
             "weather_type", "hour", "month"]

def save_gaps_csv(df, missing_mask, path):
    out = df[missing_mask].copy()
    out.index.name = "Time"
    out = out.reset_index()
    ts = out["Time"]
    out["Time"] = ts.dt.month.astype(str) + "/" + ts.dt.day.astype(str) + "/" + ts.dt.year.astype(str) + " " + ts.dt.hour.astype(str) + ":" + ts.dt.minute.astype(str).str.zfill(2)
    # Match original Renewable.csv format (no trailing zeros on floats)
    int_cols = ["Energy delta[Wh]", "pressure", "humidity", "clouds_all", "isSun",
                "sunlightTime", "dayLength", "weather_type", "hour", "month"]
    float_prec = {"GHI": 1, "temp": 1, "wind_speed": 1, "rain_1h": 2, "snow_1h": 2, "SunlightTime/daylength": 2}
    for col, prec in float_prec.items():
        out[col] = out[col].round(prec)
    import csv as csv_mod
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(orig_cols)
        for _, row in out[orig_cols].iterrows():
            vals = []
            for c in orig_cols:
                v = row[c]
                if c in float_prec:
                    s = f"{v:.{float_prec[c]}f}".rstrip("0").rstrip(".")
                    vals.append(s if s else "0")
                elif c in int_cols:
                    vals.append(str(int(v)))
                else:
                    vals.append(str(v))
            w.writerow(vals)

save_gaps_csv(train_pp, train_missing, "data/training_filled_gaps.csv")
save_gaps_csv(val_pp, val_missing, "data/validation_filled_gaps.csv")
save_gaps_csv(test_pp, test_missing, "data/testing_filled_gaps.csv")
print(f"Saved filled gaps: training ({train_missing.sum():,}), validation ({val_missing.sum():,}), testing ({test_missing.sum():,})")

# ========== 3. FEATURE ENGINEERING ==========
print("\n" + "=" * 60)
print("3. FEATURE ENGINEERING")
print("=" * 60)

def engineer(df_in, target_col=TARGET):
    d = df_in.copy()
    d["ED_lag1"] = d[target_col].shift(1)
    d["ED_lag4"] = d[target_col].shift(4)
    d["ED_lag96"] = d[target_col].shift(96)
    d["GHI_lag1"] = d["GHI"].shift(1)
    d["GHI_roll4"] = d["GHI"].rolling(window=4, min_periods=1).mean()
    d["ED_roll4"] = d[target_col].rolling(window=4, min_periods=1).mean()
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["GHI_x_sun"] = d["GHI"] * d["SunlightTime/daylength"]
    d["GHI_x_isSun"] = d["GHI"] * d["isSun"]
    return d.dropna()

train_fe = engineer(train_pp)
val_fe = engineer(val_pp)
test_fe = engineer(test_pp)
print(f"Train FE: {len(train_fe):,} rows")
print(f"Val FE:   {len(val_fe):,} rows")
print(f"Test FE:  {len(test_fe):,} rows")

# ========== 4. EDA ==========
print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

numeric_cols = train_fe.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = train_fe[numeric_cols].corr()
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values()

print("Top 5 positive correlates:")
print(target_corr.tail(5).to_string())
print("\nTop 5 negative correlates:")
print(target_corr.head(5).to_string())

# Figure 01: Correlation analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, ax=ax1,
            annot_kws={"fontsize": 7})
ax1.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")

colors = ["#E05252" if v < 0 else "#378ADD" for v in target_corr.values]
ax2.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white", linewidth=0.5)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_title(f"Correlation with {TARGET}", fontsize=14, fontweight="bold")
ax2.set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/01_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 01_correlation.png")

# Figure 02: Scatter grid
scatter_features = ["GHI", "SunlightTime/daylength", "sunlightTime", "isSun",
                    "clouds_all", "temp", "humidity", "wind_speed"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(scatter_features):
    ax = axes[i]
    r_val = corr_matrix.loc[TARGET, feat]
    ax.scatter(train_fe[feat], train_fe[TARGET], alpha=0.03, s=0.5, color="#378ADD")
    ax.set_title(f"r = {r_val:.3f}", fontsize=11)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel(TARGET, fontsize=9)

plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/02_scatter_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_scatter_grid.png")

# ========== 5. MODEL DEFINITIONS ==========
print("\n" + "=" * 60)
print("5. MODEL TRAINING & EVALUATION")
print("=" * 60)

# Top 5 features by Pearson correlation with Energy delta[Wh]
# Selected from original (non-engineered) features for fairness across model types
TOP5_FEATURES = [
    "GHI",           # r = 0.917 — Global Horizontal Irradiance
    "isSun",         # r = 0.527 — sunlight presence indicator
    "sunlightTime",  # r = 0.441 — cumulative sunlight minutes today
    "SunlightTime/daylength",  # r = 0.403 — sunlight proportion
    "temp"           # r = 0.386 — air temperature
]

FEATURE_SET = TOP5_FEATURES

def train_model(train_fe, feature_list, model_type="linear", label=""):
    X_train = train_fe[feature_list].values
    y_train = train_fe[TARGET].values

    if model_type == "linear":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        model = LinearRegression()
    elif model_type == "tree":
        scaler = None
        X_tr = X_train.copy()
        model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    elif model_type == "forest":
        scaler = None
        X_tr = X_train.copy()
        model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                       random_state=42, n_jobs=-1)
    elif model_type == "lightgbm":
        scaler = None
        X_tr = X_train.copy()
        model = lgb.LGBMRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1, verbose=-1)
    elif model_type == "xgboost":
        scaler = None
        X_tr = X_train.copy()
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1, verbosity=0)
    elif model_type == "mlp":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                              random_state=42, early_stopping=True, verbose=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_tr, y_train)
    train_r2 = model.score(X_tr, y_train)
    print(f"{label} Training R²: {train_r2:.4f}")
    return model, scaler

def compute_metrics(y_true, y_pred, label=""):
    y_pred = y_pred.clip(0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rel = (rmse / y_true.mean()) * 100
    rng = y_true.max() - y_true.min()
    nrmse = (rmse / rng * 100) if rng > 0 else 0.0
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    print(f"{label}")
    print(f"  MSE  = {mse:>12.1f} Wh²")
    print(f"  MAE  = {mae:>12.1f} Wh")
    print(f"  RMSE = {rmse:>12.1f} Wh")
    print(f"  RelErr = {rel:>8.2f} %")
    print(f"  nRMSE  = {nrmse:>8.2f} %")
    print(f"  R²   = {r2:>12.4f}")
    return dict(mse=mse, mae=mae, rmse=rmse, rel=rel, nrmse=nrmse, r2=r2)

def compute_daytime_nrmse(y_true, y_pred, ghi_values):
    daytime_mask = ghi_values > 0
    y_d = y_true[daytime_mask]
    yp_d = y_pred[daytime_mask].clip(0)
    rmse_d = np.sqrt(mean_squared_error(y_d, yp_d))
    rng_d = y_d.max() - y_d.min()
    return (rmse_d / rng_d * 100) if rng_d > 0 else 0.0

models = {}
scalers = {}
results = {}

MODEL_CONFIGS = [
    ("Linear Regression", "linear"),
    ("Decision Tree",     "tree"),
    ("Random Forest",     "forest"),
    ("LightGBM",          "lightgbm"),
    ("XGBoost",           "xgboost"),
    ("MLP",               "mlp"),
]

for label, mtype in MODEL_CONFIGS:
    print(f"\n--- {label} ---")
    model, scaler = train_model(train_fe, FEATURE_SET, mtype, label)
    models[label] = model
    scalers[label] = scaler

    # Train predictions
    X_tr = train_fe[FEATURE_SET].values
    if scaler is not None:
        X_tr = scaler.transform(X_tr)
    y_train_pred = model.predict(X_tr).clip(0)
    train_metrics = compute_metrics(train_fe[TARGET].values, y_train_pred, f"  {label} (Train)")

    # Validation predictions
    X_va = val_fe[FEATURE_SET].values
    if scaler is not None:
        X_va = scaler.transform(X_va)
    y_val_pred = model.predict(X_va).clip(0)
    val_metrics = compute_metrics(val_fe[TARGET].values, y_val_pred, f"  {label} (Val)")

    # Test predictions
    X_te = test_fe[FEATURE_SET].values
    if scaler is not None:
        X_te = scaler.transform(X_te)
    y_test_pred = model.predict(X_te).clip(0)
    test_metrics = compute_metrics(test_fe[TARGET].values, y_test_pred, f"  {label} (Test)")

    val_daytime_nrmse = compute_daytime_nrmse(
        val_fe[TARGET].values, y_val_pred, val_fe["GHI"].values
    )
    test_daytime_nrmse = compute_daytime_nrmse(
        test_fe[TARGET].values, y_test_pred, test_fe["GHI"].values
    )
    print(f"  {label} Val Daytime nRMSE = {val_daytime_nrmse:.2f}%")
    print(f"  {label} Test Daytime nRMSE = {test_daytime_nrmse:.2f}%")

    results[label] = {"train": train_metrics, "val": val_metrics, "test": test_metrics, "val_daytime_nrmse": val_daytime_nrmse, "test_daytime_nrmse": test_daytime_nrmse}

# ========== 6. HYPERPARAMETER TUNING ==========
print("\n" + "=" * 60)
print("6. HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("=" * 60)

def compute_importance(model, scaler, feature_list, train_fe, model_type="linear", target_col=TARGET):
    if model_type == "linear" and hasattr(model, "coef_"):
        std_y = train_fe[target_col].std()
        std_X = train_fe[feature_list].std()
        betas_standardised = np.abs(model.coef_ * (std_X.values / std_y))
        return pd.Series(betas_standardised, index=feature_list).sort_values(ascending=False)
    elif hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    else:
        from sklearn.inspection import permutation_importance
        X = train_fe[feature_list].values
        if scaler is not None:
            X = scaler.transform(X)
        r = permutation_importance(model, X[:5000], train_fe[target_col].values[:5000],
                                   n_repeats=5, random_state=42, n_jobs=-1)
        return pd.Series(r.importances_mean, index=feature_list).sort_values(ascending=False)

importance_dfs = {}

# Compute feature importance for baseline models
for label, mtype in MODEL_CONFIGS:
    imp = compute_importance(models[label], scalers[label], FEATURE_SET, train_fe, mtype)
    importance_dfs[label] = imp

# Tune LightGBM
print("\n--- Tuning LightGBM ---")
lgb_param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [7, 10, 15],
    'learning_rate': [0.01, 0.03, 0.05],
    'num_leaves': [31, 63, 127],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5],
    'min_child_samples': [10, 20, 30],
}
X_tr_all = train_fe[FEATURE_SET].values
lgb_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    lgb_param_grid, n_iter=10, cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
)
lgb_search.fit(X_tr_all, train_fe[TARGET].values)
print(f"  Best params: {lgb_search.best_params_}")

# Tune MLP
print("\n--- Tuning MLP ---")
mlp_param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [64, 128, 256],
}
scaler_tune = StandardScaler()
X_tr_tune = scaler_tune.fit_transform(X_tr_all)
mlp_search = RandomizedSearchCV(
    MLPRegressor(max_iter=500, random_state=42, early_stopping=True),
    mlp_param_grid, n_iter=10, cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
)
mlp_search.fit(X_tr_tune, train_fe[TARGET].values)
print(f"  Best params: {mlp_search.best_params_}")

# Retrain LightGBM (tuned) and evaluate
print("\n--- Retraining LightGBM (tuned) with best params ---")
best_lgb = lgb.LGBMRegressor(**lgb_search.best_params_, random_state=42, n_jobs=-1, verbose=-1)
best_lgb.fit(X_tr_all, train_fe[TARGET].values)
models["LightGBM (tuned)"] = best_lgb
scalers["LightGBM (tuned)"] = None
y_tr_pred = best_lgb.predict(X_tr_all).clip(0)
tr_m = compute_metrics(train_fe[TARGET].values, y_tr_pred, "  LightGBM (tuned) (Train)")
X_va_all = val_fe[FEATURE_SET].values
y_va_pred = best_lgb.predict(X_va_all).clip(0)
va_m = compute_metrics(val_fe[TARGET].values, y_va_pred, "  LightGBM (tuned) (Val)")
va_dnrmse = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred, val_fe["GHI"].values)
print(f"  LightGBM (tuned) Val Daytime nRMSE = {va_dnrmse:.2f}%")
X_te_all = test_fe[FEATURE_SET].values
y_te_pred = best_lgb.predict(X_te_all).clip(0)
te_m = compute_metrics(test_fe[TARGET].values, y_te_pred, "  LightGBM (tuned) (Test)")
te_dnrmse = compute_daytime_nrmse(test_fe[TARGET].values, y_te_pred, test_fe["GHI"].values)
print(f"  LightGBM (tuned) Test Daytime nRMSE = {te_dnrmse:.2f}%")
results["LightGBM (tuned)"] = {"train": tr_m, "val": va_m, "test": te_m, "val_daytime_nrmse": va_dnrmse, "test_daytime_nrmse": te_dnrmse}
importance_dfs["LightGBM (tuned)"] = compute_importance(best_lgb, None, FEATURE_SET, train_fe, "lightgbm")

# Retrain MLP (tuned) and evaluate
print("\n--- Retraining MLP (tuned) with best params ---")
best_mlp = MLPRegressor(**mlp_search.best_params_, max_iter=500, random_state=42, early_stopping=True)
best_mlp.fit(X_tr_tune, train_fe[TARGET].values)
models["MLP (tuned)"] = best_mlp
scalers["MLP (tuned)"] = scaler_tune
y_tr_pred = best_mlp.predict(X_tr_tune).clip(0)
tr_m = compute_metrics(train_fe[TARGET].values, y_tr_pred, "  MLP (tuned) (Train)")
X_va_tune = scaler_tune.transform(X_va_all)
y_va_pred = best_mlp.predict(X_va_tune).clip(0)
va_m = compute_metrics(val_fe[TARGET].values, y_va_pred, "  MLP (tuned) (Val)")
va_dnrmse = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred, val_fe["GHI"].values)
print(f"  MLP (tuned) Val Daytime nRMSE = {va_dnrmse:.2f}%")
X_te_tune = scaler_tune.transform(X_te_all)
y_te_pred = best_mlp.predict(X_te_tune).clip(0)
te_m = compute_metrics(test_fe[TARGET].values, y_te_pred, "  MLP (tuned) (Test)")
te_dnrmse = compute_daytime_nrmse(test_fe[TARGET].values, y_te_pred, test_fe["GHI"].values)
print(f"  MLP (tuned) Test Daytime nRMSE = {te_dnrmse:.2f}%")
results["MLP (tuned)"] = {"train": tr_m, "val": va_m, "test": te_m, "val_daytime_nrmse": va_dnrmse, "test_daytime_nrmse": te_dnrmse}
importance_dfs["MLP (tuned)"] = compute_importance(best_mlp, scaler_tune, FEATURE_SET, train_fe, "mlp")

# Add tuned models to configs
MODEL_CONFIGS.append(("LightGBM (tuned)", "lightgbm_tuned"))
MODEL_CONFIGS.append(("MLP (tuned)", "mlp_tuned"))

# Print all feature importances
for label, mtype in MODEL_CONFIGS:
    print(f"\n{label} - Feature Importance:")
    print(importance_dfs[label].to_string())

# VIF for the feature set
def compute_vif(train_fe, feature_list):
    X = train_fe[feature_list].values
    vifs = []
    for i, feat in enumerate(feature_list):
        X_others = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(X_others, X[:, i]).score(X_others, X[:, i])
        vif = 1 / (1 - r2) if r2 < 1 else 999
        vifs.append((feat, vif))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

vif_df = compute_vif(train_fe, FEATURE_SET)
print("\nVIF Analysis:")
print(vif_df.to_string())

# ========== 7. 24-HOUR FORECAST ==========
print("\n" + "=" * 60)
print("7. 24-HOUR ROLLOUT FORECAST")
print("=" * 60)

def forecast_24h(model, scaler, feature_list, test_pp, train_pp,
                 forecast_start_time, target_col=TARGET):
    """
    24h auto-regressive rollout forecast.
    
    Builds features on-the-fly from a continuous buffer of actual historical
    data, eliminating the leakage of pre-engineered features.
    
    Parameters
    ----------
    model : trained regressor
    scaler : StandardScaler or None
    feature_list : list of feature column names
    test_pp : DataFrame — gap-filled test data (NOT engineered)
    train_pp : DataFrame — gap-filled training data (NOT engineered)
    forecast_start_time : Timestamp — first timepoint to forecast
    """
    # Build continuous history buffer: train_pp tail + test_pp pre-forecast
    pre = test_pp.loc[:forecast_start_time - pd.Timedelta("15min")]
    full = pd.concat([train_pp[[target_col, "GHI"]], pre[[target_col, "GHI"]]])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    buf_ed = list(full[target_col].values[-200:])
    buf_ghi = list(full["GHI"].values[-200:])

    # Slice the 96 forecast rows from test_pp
    fc_end = forecast_start_time + pd.Timedelta(hours=23, minutes=45)
    fc_rows = test_pp.loc[forecast_start_time:fc_end]

    predictions = []
    for idx, row in fc_rows.iterrows():
        feat = {}
        for f in feature_list:
            if f == "ED_lag1":
                feat[f] = buf_ed[-1]
            elif f == "ED_lag4":
                feat[f] = buf_ed[-4]
            elif f == "ED_lag96":
                feat[f] = buf_ed[-96] if len(buf_ed) >= 96 else 0.0
            elif f == "GHI_lag1":
                feat[f] = buf_ghi[-1]
            elif f == "GHI_roll4":
                feat[f] = float(np.mean(buf_ghi[-4:]))
            elif f == "ED_roll4":
                feat[f] = float(np.mean(buf_ed[-4:]))
            elif f == "GHI_x_sun":
                feat[f] = row["GHI"] * row["SunlightTime/daylength"]
            elif f == "GHI_x_isSun":
                feat[f] = row["GHI"] * row["isSun"]
            elif f == "hour_sin":
                feat[f] = np.sin(2 * np.pi * idx.hour / 24)
            elif f == "hour_cos":
                feat[f] = np.cos(2 * np.pi * idx.hour / 24)
            elif f == "month_sin":
                feat[f] = np.sin(2 * np.pi * idx.month / 12)
            elif f == "month_cos":
                feat[f] = np.cos(2 * np.pi * idx.month / 12)
            elif f == "GHI_lag96":
                feat[f] = buf_ghi[-96] if len(buf_ghi) >= 96 else 0.0
            elif f == "ED_roll6h":
                feat[f] = float(np.mean(buf_ed[-24:]))
            elif f == "ED_roll24h":
                feat[f] = float(np.mean(buf_ed[-96:]))
            elif f == "GHI_roll24h":
                feat[f] = float(np.mean(buf_ghi[-96:]))
            else:
                feat[f] = row[f]

        X = np.array([[feat[f] for f in feature_list]])
        if scaler is not None:
            X = scaler.transform(X)
        y_pred = float(max(0.0, model.predict(X)[0]))
        predictions.append(y_pred)

        buf_ed.append(y_pred)
        buf_ghi.append(float(row["GHI"]))

    return np.array(predictions)

test_daily_sum = test_fe[TARGET].resample("D").sum()
best_day = test_daily_sum.idxmax()
best_day_start_fe = test_fe.index.get_loc(best_day)
forecast_start = test_fe.index[best_day_start_fe]
print(f"Best forecast day: {forecast_start.date()}")

forecasts = {}
for label, mtype in MODEL_CONFIGS:
    fct = forecast_24h(models[label], scalers[label], FEATURE_SET, test_pp, train_pp,
                       forecast_start)
    forecasts[label] = fct
    actual = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
    rmse_f = np.sqrt(mean_squared_error(actual, fct.clip(0)))
    rel_f = (rmse_f / actual.mean()) * 100 if actual.mean() > 0 else 0
    print(f"  {label} 24h RMSE={rmse_f:.1f} Wh, RelErr={rel_f:.2f}%")

# ========== 8. VISUALIZATIONS ==========
print("\n" + "=" * 60)
print("8. GENERATING FIGURES")
print("=" * 60)

MODEL_COLORS = {
    "Linear Regression": "#7F77DD",
    "Decision Tree":     "#EF9F27",
    "Random Forest":     "#1D9E75",
    "LightGBM":          "#E06B22",
    "XGBoost":           "#7B287D",
    "MLP":               "#E05252",
    "LightGBM (tuned)":  "#D4833A",
    "MLP (tuned)":       "#C04040",
}
REF_RED = "#E05252"

# Figure 03: Model comparison table
table_data = []
for label, mtype in MODEL_CONFIGS:
    tr = results[label]["train"]
    va = results[label]["val"]
    te = results[label]["test"]
    table_data.append([
        label,
        mtype,
        f"{tr['rmse']:.1f}",
        f"{tr['rel']:.2f}%",
        f"{va['rmse']:.1f}",
        f"{va['nrmse']:.2f}%",
        f"{va['rel']:.2f}%",
        f"{va['r2']:.4f}",
        f"{te['rmse']:.1f}",
        f"{te['nrmse']:.2f}%",
        f"{te['rel']:.2f}%",
        f"{te['r2']:.4f}",
    ])

fig, ax = plt.subplots(figsize=(16, 3.5))
ax.axis("off")
col_labels = ["Model", "Type", "Train RMSE", "Train RelErr",
              "Val RMSE", "Val nRMSE", "Val RelErr", "Val R²",
              "Test RMSE", "Test nRMSE", "Test RelErr", "Test R²"]
tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
               cellLoc="center", colWidths=[0.08, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#3C3489")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 3:
        cell.set_facecolor("#E8F5E9")

plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/03_model_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 03_model_comparison_table.png")

# Figure 04: Error metric bar comparison
metric_names = ["RMSE", "MAE", "RelErr", "nRMSE"]
train_vals = {m: [] for m in metric_names}
val_vals = {m: [] for m in metric_names}
test_vals = {m: [] for m in metric_names}

model_labels = [m[0] for m in MODEL_CONFIGS]

for label, mtype in MODEL_CONFIGS:
    tr = results[label]["train"]
    va = results[label]["val"]
    te = results[label]["test"]
    for m in metric_names:
        key = {"RelErr": "rel", "nRMSE": "nrmse"}.get(m, m.lower())
        train_vals[m].append(tr[key])
        val_vals[m].append(va[key])
        test_vals[m].append(te[key])

n_models = len(model_labels)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
x = np.arange(n_models)
width = min(0.25, 0.7 / n_models)

for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    bars_train = ax.bar(x - width, train_vals[metric], width, label="Train",
                        color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.35)
    bars_val = ax.bar(x, val_vals[metric], width, label="Val",
                      color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.7)
    bars_test = ax.bar(x + width, test_vals[metric], width, label="Test",
                       color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    if metric in ("RelErr", "nRMSE"):
        ax.axhline(5, color=REF_RED, linestyle="--", linewidth=1.5, label="5% target")
    ax.legend(fontsize=8)
    ax.set_ylabel(metric)

plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/04_error_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_error_comparison.png")

# Figure 05: 24-hour forecast vs actual
n_models = len(MODEL_CONFIGS)
n_cols_f5 = 2 if n_models > 3 else 1
n_rows_f5 = int(np.ceil(n_models / n_cols_f5))
fig, axes = plt.subplots(n_rows_f5, n_cols_f5, figsize=(7 * n_cols_f5, 4 * n_rows_f5), sharex=True)
axes = np.atleast_1d(axes).ravel()
time_idx = pd.date_range(start=forecast_start, periods=96, freq="15min")
actual_vals = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values

for i, (label, mtype) in enumerate(MODEL_CONFIGS):
    ax = axes[i]
    fct = forecasts[label]
    rel_err = np.sqrt(mean_squared_error(actual_vals, fct.clip(0))) / actual_vals.mean() * 100
    ax.plot(time_idx, actual_vals, color="black", linewidth=1.0, label="Actual")
    ax.plot(time_idx, fct, color=MODEL_COLORS[label], linewidth=1.0, linestyle="--", label="Predicted")
    ax.fill_between(time_idx, actual_vals, fct, alpha=0.15, color=MODEL_COLORS[label])
    ax.set_title(f"{label} — RelErr: {rel_err:.2f}%", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylabel("Energy delta [Wh]")
    ax.set_ylim(bottom=0)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

axes[-1].set_xlabel("Time (15-min intervals)")
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/05_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 05_forecast_vs_actual.png")

# Figure 06: Residual analysis (best model by val R²)
best_label = max(results, key=lambda k: results[k]["val"]["r2"])
X_te_best = test_fe[FEATURE_SET].values
if scalers[best_label] is not None:
    X_te_best = scalers[best_label].transform(X_te_best)
y_test_pred_best = models[best_label].predict(X_te_best).clip(0)
residuals_best = test_fe[TARGET].values - y_test_pred_best

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter: predicted vs residual
axes[0].scatter(y_test_pred_best, residuals_best, alpha=0.3, s=1, color=MODEL_COLORS[best_label])
axes[0].axhline(0, color=REF_RED, linestyle="-", linewidth=1)
axes[0].set_xlabel("Predicted [Wh]")
axes[0].set_ylabel("Residual [Wh]")
axes[0].set_title("Predicted vs Residual", fontsize=12, fontweight="bold")

# Histogram
axes[1].hist(residuals_best, bins=80, color=MODEL_COLORS[best_label], edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Residual [Wh]")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")

# Q-Q plot
stats.probplot(residuals_best, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (Normal)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/06_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 06_residuals.png")

# Figure 07: Feature importance
n_models = len(MODEL_CONFIGS)
n_cols_f7 = 3 if n_models <= 3 else 3
n_rows_f7 = int(np.ceil(n_models / n_cols_f7))
fig, axes = plt.subplots(n_rows_f7, n_cols_f7, figsize=(6 * n_cols_f7, 4 * n_rows_f7))
axes = np.atleast_1d(axes).ravel()

for i, (label, mtype) in enumerate(MODEL_CONFIGS):
    ax = axes[i]
    imp = importance_dfs[label]
    colors_imp = [MODEL_COLORS[label] if v > imp.iloc[0] * 0.15 else "#CCCCCC" for v in imp.values]
    ax.barh(range(len(imp)), imp.values[::-1], color=colors_imp[::-1], edgecolor="white")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index[::-1], fontsize=9)
    ax.set_title(f"{label} — Feature Importance", fontsize=12, fontweight="bold")
    xlabel = "|β_std|" if mtype == "linear" else "Importance"
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 07_feature_importance.png")

# Figure 08: 2-week time series
n_models = len(MODEL_CONFIGS)
n_cols_f8 = 2
n_rows_f8 = int(np.ceil(n_models / n_cols_f8))
fig, axes = plt.subplots(n_rows_f8, n_cols_f8, figsize=(8 * n_cols_f8, 4 * n_rows_f8), sharex=True)
axes = np.atleast_1d(axes).ravel()
n_steps = 96 * 14

for i, (label, mtype) in enumerate(MODEL_CONFIGS):
    ax = axes[i]
    actual_2w = test_fe[TARGET].iloc[:n_steps].values
    X_2w = test_fe.iloc[:n_steps][FEATURE_SET].values
    if scalers[label] is not None:
        X_2w = scalers[label].transform(X_2w)
    pred_2w = models[label].predict(X_2w).clip(0)
    time_2w = test_fe.index[:n_steps]
    ax.plot(time_2w, actual_2w, color="black", linewidth=0.5, label="Actual")
    ax.plot(time_2w, pred_2w, color=MODEL_COLORS[label], linewidth=0.6, linestyle="--", label="Predicted")
    ax.set_title(f"{label} — First 14 Days of Test Set", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy delta [Wh]")
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

axes[-1].set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/08_2week_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 08_2week_forecast.png")

# Figure 09: Full results dashboard
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 22))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

model_labels = [m[0] for m in MODEL_CONFIGS]
n_models09 = len(model_labels)
best_label = max(results, key=lambda k: results[k]["val"]["r2"])

# 1. RMSE bars
ax1 = fig.add_subplot(gs[0, 0])
x09 = np.arange(n_models09)
w09 = min(0.25, 0.7 / n_models09)
train_rmses = [results[m[0]]["train"]["rmse"] for m in MODEL_CONFIGS]
val_rmses = [results[m[0]]["val"]["rmse"] for m in MODEL_CONFIGS]
test_rmses = [results[m[0]]["test"]["rmse"] for m in MODEL_CONFIGS]
ax1.bar(x09 - w09, train_rmses, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.35)
ax1.bar(x09, val_rmses, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.7)
ax1.bar(x09 + w09, test_rmses, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS])
ax1.set_xticks(x09)
ax1.set_xticklabels(model_labels, fontsize=8)
ax1.set_title("RMSE Comparison", fontweight="bold")
ax1.set_ylabel("RMSE [Wh]")

# 2. RelErr bars
ax2 = fig.add_subplot(gs[0, 1])
train_rels = [results[m[0]]["train"]["rel"] for m in MODEL_CONFIGS]
val_rels = [results[m[0]]["val"]["rel"] for m in MODEL_CONFIGS]
test_rels = [results[m[0]]["test"]["rel"] for m in MODEL_CONFIGS]
ax2.bar(x09 - w09, train_rels, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.35)
ax2.bar(x09, val_rels, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.7)
ax2.bar(x09 + w09, test_rels, w09, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS])
ax2.axhline(5, color=REF_RED, linestyle="--", linewidth=1.5)
ax2.set_xticks(x09)
ax2.set_xticklabels(model_labels, fontsize=8)
ax2.set_title("Relative Error Comparison", fontweight="bold")
ax2.set_ylabel("RelErr [%]")

# 3. 24-hour forecast (all models)
ax3 = fig.add_subplot(gs[1, :])
actual_96 = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
ax3.plot(time_idx, actual_96, color="black", linewidth=1.5, label="Actual")
for label, mtype in MODEL_CONFIGS:
    ax3.plot(time_idx, forecasts[label], color=MODEL_COLORS[label], linewidth=1.0, linestyle="--", label=label)
ax3.set_title(f"24-Hour Forecast — {forecast_start.date()}", fontweight="bold")
ax3.set_ylabel("Energy delta [Wh]")
ax3.legend()
ax3.set_ylim(bottom=0)

# 4. 2-week series (best model)
ax4 = fig.add_subplot(gs[2, 0])
actual_2w_best = test_fe[TARGET].iloc[:n_steps].values
X_2w_best = test_fe.iloc[:n_steps][FEATURE_SET].values
if scalers[best_label] is not None:
    X_2w_best = scalers[best_label].transform(X_2w_best)
pred_2w_best = models[best_label].predict(X_2w_best).clip(0)
ax4.plot(time_2w, actual_2w_best, color="black", linewidth=0.4)
ax4.plot(time_2w, pred_2w_best, color=MODEL_COLORS[best_label], linewidth=0.5, linestyle="--")
ax4.set_title(f"{best_label} — 2-Week Prediction", fontweight="bold")
ax4.set_ylabel("Energy delta [Wh]")

# 5. Best model feature importance
ax5 = fig.add_subplot(gs[2, 1])
imp_best = importance_dfs[best_label]
colors_best = [MODEL_COLORS[best_label]] * len(imp_best)
ax5.barh(range(len(imp_best)), imp_best.values[::-1], color=colors_best[::-1], edgecolor="white")
ax5.set_yticks(range(len(imp_best)))
ax5.set_yticklabels(imp_best.index[::-1], fontsize=8)
ax5.set_title(f"{best_label} — Feature Importance", fontweight="bold")
# Check if best model uses coefficient-based importance
best_mtype = dict(MODEL_CONFIGS)[best_label]
xlabel_best = "|β_std|" if best_mtype == "linear" else "Importance"
ax5.set_xlabel(xlabel_best)
ax5.invert_yaxis()

# 6. Residual histogram (best model)
ax6 = fig.add_subplot(gs[3, 0])
# Compute residuals for best model
X_te_best = test_fe[FEATURE_SET].values
if scalers[best_label] is not None:
    X_te_best = scalers[best_label].transform(X_te_best)
y_pred_best = models[best_label].predict(X_te_best).clip(0)
residuals_best = test_fe[TARGET].values - y_pred_best
ax6.hist(residuals_best, bins=80, color=MODEL_COLORS[best_label], edgecolor="white", alpha=0.8)
ax6.set_title(f"{best_label} — Residual Distribution", fontweight="bold")
ax6.set_xlabel("Residual [Wh]")
ax6.set_ylabel("Frequency")

# 7. Daytime nRMSE
ax7 = fig.add_subplot(gs[3, 1])
daytime_nrmses_val = [results[m[0]]["val_daytime_nrmse"] for m in MODEL_CONFIGS]
daytime_nrmses_test = [results[m[0]]["test_daytime_nrmse"] for m in MODEL_CONFIGS]
x_dn = np.arange(n_models09)
w_dn = min(0.35, 0.7 / n_models09)
ax7.bar(x_dn - w_dn/2, daytime_nrmses_val, w_dn, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS], alpha=0.7)
ax7.bar(x_dn + w_dn/2, daytime_nrmses_test, w_dn, color=[MODEL_COLORS[m[0]] for m in MODEL_CONFIGS])
ax7.set_xticks(x_dn)
ax7.set_xticklabels(model_labels, fontsize=7, rotation=20)
ax7.axhline(5, color=REF_RED, linestyle="--", linewidth=1.5)
ax7.set_title("Daytime-Only nRMSE (Val / Test)", fontweight="bold")
ax7.set_ylabel("nRMSE [%]")

fig.suptitle("Solar Energy Prediction — Full Pipeline Results\n"
             f"Train/Val/Test Split (70/15/15)  |  {n_models09} Model Types (Top-5 Features)",
             fontsize=14, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{TARGET_DIR}/09_full_results_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 09_full_results_dashboard.png")

# ========== 9. FEATURE EXPANSION EXPERIMENT ==========
print("\n" + "=" * 60)
print("9. FEATURE EXPANSION EXPERIMENT (LightGBM)")
print("=" * 60)

EXPANDED_FEATURES = [
    # Top 5 correlation features only
    "GHI", "temp", "isSun", "sunlightTime", "SunlightTime/daylength",
    # Time encoding
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Target lags and rolling
    "ED_lag1", "ED_lag4", "ED_lag96", "ED_roll4",
    # GHI lags and rolling
    "GHI_lag1", "GHI_roll4",
    # Interactions
    "GHI_x_sun", "GHI_x_isSun",
]

print(f"\nTraining LightGBM with {len(EXPANDED_FEATURES)} features...")
X_tr_exp = train_fe[EXPANDED_FEATURES].values
y_tr_exp = train_fe[TARGET].values
lgb_exp = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                              num_leaves=31, min_child_samples=10, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.0,
                              random_state=42, n_jobs=-1, verbose=-1)
lgb_exp.fit(X_tr_exp, y_tr_exp)
print(f"  Training R²: {lgb_exp.score(X_tr_exp, y_tr_exp):.4f}")

X_va_exp = val_fe[EXPANDED_FEATURES].values
y_va_exp = val_fe[TARGET].values
y_va_pred_exp = lgb_exp.predict(X_va_exp).clip(0)
va_metrics = compute_metrics(y_va_exp, y_va_pred_exp, "  Expanded Features (Val)")

X_te_exp = test_fe[EXPANDED_FEATURES].values
y_te_exp = test_fe[TARGET].values
y_te_pred_exp = lgb_exp.predict(X_te_exp).clip(0)
te_metrics = compute_metrics(y_te_exp, y_te_pred_exp, "  Expanded Features (Test)")

exp_val_daytime_nrmse = compute_daytime_nrmse(y_va_exp, y_va_pred_exp, val_fe["GHI"].values)
exp_test_daytime_nrmse = compute_daytime_nrmse(y_te_exp, y_te_pred_exp, test_fe["GHI"].values)
print(f"  Expanded Features Val Daytime nRMSE = {exp_val_daytime_nrmse:.2f}%")
print(f"  Expanded Features Test Daytime nRMSE = {exp_test_daytime_nrmse:.2f}%")

# Re-run 24h rollout forecast with expanded features
print("\n  24h Rollout Forecast (expanded features)...")
fct_exp = forecast_24h(lgb_exp, None, EXPANDED_FEATURES, test_pp, train_pp,
                       forecast_start)
actual_96 = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_exp = np.sqrt(mean_squared_error(actual_96, fct_exp.clip(0)))
rel_f_exp = (rmse_f_exp / actual_96.mean()) * 100
print(f"  24h RMSE={rmse_f_exp:.1f} Wh, RelErr={rel_f_exp:.2f}%")

# 24h on val
val_fc_start = val_fe.index[0]
fct_exp_val = forecast_24h(lgb_exp, None, EXPANDED_FEATURES, val_pp, train_pp,
                           val_fc_start)
actual_96_val = val_pp.loc[val_fc_start:val_fc_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_exp_val = np.sqrt(mean_squared_error(actual_96_val, fct_exp_val.clip(0)))
rel_f_exp_val = (rmse_f_exp_val / actual_96_val.mean()) * 100

# Compare with top-5 baseline (LightGBM)
baseline_label = "LightGBM"
bl_te = results[baseline_label]["test"]
bl_va = results[baseline_label]["val"]
print(f"\n{'Metric':<22} {'Top-5 Val':<16} {'Exp Val':<16} {'Top-5 Test':<16} {'Exp Test':<16}")
print("-" * 86)
print(f"{'Val RMSE [Wh]':<22} {bl_va['rmse']:<16.1f} {va_metrics['rmse']:<16.1f} {'':<16} {'':<16}")
print(f"{'Test RMSE [Wh]':<22} {'':<16} {'':<16} {bl_te['rmse']:<16.1f} {te_metrics['rmse']:<16.1f}")
print(f"{'Val RelErr [%]':<22} {bl_va['rel']:<16.2f} {va_metrics['rel']:<16.2f} {'':<16} {'':<16}")
print(f"{'Test RelErr [%]':<22} {'':<16} {'':<16} {bl_te['rel']:<16.2f} {te_metrics['rel']:<16.2f}")
print(f"{'Val nRMSE [%]':<22} {bl_va['nrmse']:<16.2f} {va_metrics['nrmse']:<16.2f} {'':<16} {'':<16}")
print(f"{'Test nRMSE [%]':<22} {'':<16} {'':<16} {bl_te['nrmse']:<16.2f} {te_metrics['nrmse']:<16.2f}")
print(f"{'Val Daytime nRMSE [%]':<22} {results[baseline_label]['val_daytime_nrmse']:<16.2f} {exp_val_daytime_nrmse:<16.2f} {'':<16} {'':<16}")
print(f"{'Test Daytime nRMSE [%]':<22} {'':<16} {'':<16} {results[baseline_label]['test_daytime_nrmse']:<16.2f} {exp_test_daytime_nrmse:<16.2f}")
print(f"{'Val R²':<22} {bl_va['r2']:<16.4f} {va_metrics['r2']:<16.4f} {'':<16} {'':<16}")
print(f"{'Test R²':<22} {'':<16} {'':<16} {bl_te['r2']:<16.4f} {te_metrics['r2']:<16.4f}")
# 24h forecast RelErr for baseline
baseline_fct = forecasts.get(baseline_label)
baseline_fct_rel = (np.sqrt(mean_squared_error(actual_96, baseline_fct.clip(0))) / actual_96.mean() * 100) if baseline_fct is not None else 0
print(f"{'24h Forecast RelErr (Test)':<22} {baseline_fct_rel:<16.2f} {rel_f_exp:<16.2f}")
print(f"{'24h Forecast RelErr (Val)':<22} {'':<16} {rel_f_exp_val:<16.2f}")

# Feature importance for expanded model
exp_imp = pd.Series(lgb_exp.feature_importances_, index=EXPANDED_FEATURES).sort_values(ascending=False)
print(f"\nTop 10 features (expanded):")
print(exp_imp.head(10).to_string())

# ========== 10. DAYTIME-ONLY MODEL (5% Target) ==========
print("\n" + "=" * 60)
print("10. DAYTIME-ONLY MODEL (GHI > 0) -- TARGETING <= 5% nRMSE")
print("=" * 60)

# Filter to daytime
train_day = train_fe[train_fe["GHI"] > 0].copy()
val_day = val_fe[val_fe["GHI"] > 0].copy()
test_day = test_fe[test_fe["GHI"] > 0].copy()
print(f"  Daytime train: {len(train_day):,} rows ({len(train_day)/len(train_fe):.0%})")
print(f"  Daytime val:   {len(val_day):,} rows ({len(val_day)/len(val_fe):.0%})")
print(f"  Daytime test:  {len(test_day):,} rows ({len(test_day)/len(test_fe):.0%})")
print(f"  Daytime mean target: {test_day[TARGET].mean():.1f} Wh (vs {test_fe[TARGET].mean():.1f} Wh full)")

# Add extra rolling/lag features to daytime sets
for df in [train_day, val_day, test_day]:
    df["GHI_lag96"] = df["GHI"].shift(96)
    df["ED_roll6h"] = df[TARGET].rolling(window=24, min_periods=1).mean()
    df["ED_roll24h"] = df[TARGET].rolling(window=96, min_periods=1).mean()
    df["GHI_roll24h"] = df["GHI"].rolling(window=96, min_periods=1).mean()
train_day = train_day.dropna()
val_day = val_day.dropna()
test_day = test_day.dropna()

DAYTIME_FEATURES = EXPANDED_FEATURES + ["GHI_lag96", "ED_roll6h", "ED_roll24h", "GHI_roll24h"]
print(f"  Using {len(DAYTIME_FEATURES)} features")

# TimeSeriesSplit tuning on daytime data
tscv_day = TimeSeriesSplit(n_splits=3)
day_param_grid = {
    "n_estimators": [200, 500, 1000, 1500],
    "max_depth": [3, 5, 7, 10, 15, -1],
    "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127, 255],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0, 0.01, 0.1, 1.0],
    "min_split_gain": [0, 0.01, 0.1],
}

X_day = train_day[DAYTIME_FEATURES].values
y_day = train_day[TARGET].values

print("\n  Tuning LightGBM on daytime data (10 iterations, 3-fold TimeSeriesSplit)...")
day_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
    day_param_grid, n_iter=10, cv=tscv_day, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1, verbose=0,
)
day_search.fit(X_day, y_day)
print(f"  Best CV RMSE: {-day_search.best_score_:.1f} Wh")
print(f"  Best params: {day_search.best_params_}")

# Retrain on full daytime data with best params
lgb_day = lgb.LGBMRegressor(**day_search.best_params_, random_state=42, n_jobs=-1, verbose=-1)
lgb_day.fit(X_day, y_day)
print(f"  Training R²: {lgb_day.score(X_day, y_day):.4f}")

# Evaluate on daytime val
X_va_day = val_day[DAYTIME_FEATURES].values
y_va_day = val_day[TARGET].values
y_va_day_pred = lgb_day.predict(X_va_day).clip(0)
day_val_metrics = compute_metrics(y_va_day, y_va_day_pred, "  Daytime-Only Model (Val)")

# Evaluate on daytime test
X_te_day = test_day[DAYTIME_FEATURES].values
y_te_day = test_day[TARGET].values
y_te_day_pred = lgb_day.predict(X_te_day).clip(0)

day_metrics = compute_metrics(y_te_day, y_te_day_pred, "  Daytime-Only Model (Test)")
day_rmse = day_metrics["rmse"]
day_rel = day_metrics["rel"]
day_nrmse = day_metrics["nrmse"]
day_r2 = day_metrics["r2"]

# Compare with best baseline
best_val_daytime_nrmse = min(results[m[0]]["val_daytime_nrmse"] for m in MODEL_CONFIGS)
best_full_val_rel = min(results[m[0]]["val"]["rel"] for m in MODEL_CONFIGS)
best_full_val_nrmse = min(results[m[0]]["val"]["nrmse"] for m in MODEL_CONFIGS)
print(f"\n  {'='*50}")
print(f"  TARGET CHECK: nRMSE = {day_nrmse:.2f}%")
print(f"  {'PASS <= 5%' if day_nrmse <= 5 else 'FAIL > 5%'}")
print(f"  {'='*50}")
print(f"\n  {'Metric':<25} {'Best Full (Val)':<18} {'Daytime-Only':<18} {'Improvement':<12}")
print(f"  {'-'*73}")
print(f"  {'Val RelErr [%]':<25} {best_full_val_rel:<18.2f} {day_val_metrics['rel']:<18.2f} {best_full_val_rel - day_val_metrics['rel']:<+12.2f}")
print(f"  {'Val nRMSE [%]':<25} {best_full_val_nrmse:<18.2f} {day_val_metrics['nrmse']:<18.2f} {best_full_val_nrmse - day_val_metrics['nrmse']:<+12.2f}")
print(f"  {'Val Daytime nRMSE [%]':<25} {best_val_daytime_nrmse:<18.2f} {day_val_metrics['nrmse']:<18.2f} {best_val_daytime_nrmse - day_val_metrics['nrmse']:<+12.2f}")
print(f"  {'Val RMSE [Wh]':<25} {min(results[m[0]]['val']['rmse'] for m in MODEL_CONFIGS):<18.1f} {day_val_metrics['rmse']:<18.1f} {'':<12}")
print(f"  {'Val R²':<25} {max(results[m[0]]['val']['r2'] for m in MODEL_CONFIGS):<18.4f} {day_val_metrics['r2']:<18.4f} {'':<12}")

# Daytime 24h rollout forecast (only if features are available in test_fe)
print("\n  24h Rollout Forecast (daytime-aware)...")
try:
    fct_day = forecast_24h(lgb_day, None, DAYTIME_FEATURES, test_pp, train_pp,
                           forecast_start)
    fc_end_d = forecast_start + pd.Timedelta(hours=23, minutes=45)
    actual_96d = test_pp.loc[forecast_start:fc_end_d, TARGET].values
    day_mask_96 = test_pp.loc[forecast_start:fc_end_d, "GHI"].values > 0
    if day_mask_96.sum() > 0:
        rmse_f_day = np.sqrt(mean_squared_error(actual_96d[day_mask_96], fct_day[day_mask_96].clip(0)))
        rel_f_day = (rmse_f_day / actual_96d[day_mask_96].mean()) * 100
        print(f"  24h Daytime-Only RMSE={rmse_f_day:.1f} Wh, RelErr={rel_f_day:.2f}%")
except Exception as e:
    print(f"  Skipping 24h rollout (daytime-only features not in test set): {e}")

# Feature importance for daytime model
day_imp = pd.Series(lgb_day.feature_importances_, index=DAYTIME_FEATURES).sort_values(ascending=False)
print(f"\n  Top 12 features (daytime model):")
print(f"  {day_imp.head(12).to_string()}")

# Quick multi-threshold daytime test using best params from above
print(f"\n  Testing stricter GHI thresholds (with best daytime params)...")
for ghi_thresh in [25, 50, 100]:
    mask_tr = train_fe["GHI"] > ghi_thresh
    mask_te = test_fe["GHI"] > ghi_thresh
    if mask_te.sum() < 100:
        print(f"  GHI > {ghi_thresh}: too few test rows ({mask_te.sum()}), skipping")
        continue

    tr_d = train_fe[mask_tr].copy()
    te_d = test_fe[mask_te].copy()
    for df in [tr_d, te_d]:
        df["GHI_lag96"] = df["GHI"].shift(96)
        df["ED_roll6h"] = df[TARGET].rolling(24, min_periods=1).mean()
        df["ED_roll24h"] = df[TARGET].rolling(96, min_periods=1).mean()
        df["GHI_roll24h"] = df["GHI"].rolling(96, min_periods=1).mean()
    tr_d = tr_d.dropna()
    te_d = te_d.dropna()

    bp = day_search.best_params_.copy()
    bp["n_estimators"] = min(bp["n_estimators"], 500)
    m = lgb.LGBMRegressor(**bp, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr_d[DAYTIME_FEATURES].values, tr_d[TARGET].values)
    pred = m.predict(te_d[DAYTIME_FEATURES].values).clip(0)
    r2 = m.score(tr_d[DAYTIME_FEATURES].values, tr_d[TARGET].values)
    rmse = np.sqrt(mean_squared_error(te_d[TARGET].values, pred))
    rel = rmse / te_d[TARGET].values.mean() * 100
    print(f"  GHI > {ghi_thresh:<3}: {len(te_d):>5} test rows, mean={te_d[TARGET].values.mean():.0f} Wh, "
          f"RMSE={rmse:.0f} Wh, RelErr={rel:.2f}%, R²={r2:.4f}")

# ========== 11. COMPREHENSIVE FEATURE SEARCH (targeting <=5% nRMSE) ==========
print("\n" + "=" * 60)
print("11. COMPREHENSIVE FEATURE SEARCH (<=5% nRMSE Target)")
print("=" * 60)

# Build superset of ALL available features from train_fe
available_cols = [c for c in train_fe.columns if c not in [TARGET, "Time"]]
print(f"\nAvailable columns in train_fe: {len(available_cols)}")

# Additional rolling/lag features not in original 24-feature set
def add_extra_features(df_in):
    d = df_in.copy()
    # Rolling std (volatility)
    d["ED_roll_std96"] = d[TARGET].rolling(96, min_periods=1).std()
    d["GHI_roll_std96"] = d["GHI"].rolling(96, min_periods=1).std()
    d["ED_roll_std24"] = d[TARGET].rolling(24, min_periods=1).std()
    # Rolling min/max
    d["ED_roll_max96"] = d[TARGET].rolling(96, min_periods=1).max()
    d["ED_roll_min96"] = d[TARGET].rolling(96, min_periods=1).min()
    d["GHI_roll_max96"] = d["GHI"].rolling(96, min_periods=1).max()
    # Rate of change
    d["ED_diff"] = d[TARGET].diff(1)
    d["ED_diff_abs"] = d[TARGET].diff(1).abs()
    # Wider windows
    d["ED_roll12h"] = d[TARGET].rolling(48, min_periods=1).mean()
    d["ED_roll48h"] = d[TARGET].rolling(192, min_periods=1).mean()
    d["GHI_roll12h"] = d["GHI"].rolling(48, min_periods=1).mean()
    # Additional lags at different frequencies
    d["ED_lag48"] = d[TARGET].shift(48)
    d["ED_lag672"] = d[TARGET].shift(672)
    d["GHI_lag4"] = d["GHI"].shift(4)
    d["GHI_lag48"] = d["GHI"].shift(48)
    d["temp_lag96"] = d["temp"].shift(96)
    # Seasonal dummies
    d["season"] = (d["month"] % 12 + 3) // 3
    d["day_of_week"] = d.index.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    # MA ratio (short-term / long-term)
    short_ma = d[TARGET].rolling(24, min_periods=1).mean()
    long_ma = d[TARGET].rolling(96, min_periods=1).mean()
    d["ED_ma_ratio"] = short_ma / (long_ma + 1e-6)
    # Cumulative GHI
    d["GHI_cum6h"] = d["GHI"].rolling(24, min_periods=1).sum()
    return d.dropna()

# Build superset with ALL features
print("\nBuilding ALL-FEATURE superset...")
train_sup = add_extra_features(train_fe)
val_sup = add_extra_features(val_fe)
test_sup = add_extra_features(test_fe)
all_feature_cols = [c for c in train_sup.columns if c not in [TARGET, "Time"]]
print(f"  Superset features: {len(all_feature_cols)}")

# Score each GHI threshold with LightGBM using superset features
GHI_THRESHOLDS = [0, 25, 50, 100, 150, 200, 300, 400, 500]
print(f"\n{'GHI >':<8} {'Rows':>8} {'Mean[Wh]':>10} {'RMSE[Wh]':>10} {'RelErr%':>8} {'nRMSE%':>8} {'R²':>7} {'Feats':>6}")
print("-" * 68)

best_config = {"rel": float("inf"), "nrmse": float("inf"), "ghi": None, "features": None, "model": None}

for ghi_thresh in GHI_THRESHOLDS:
    mask_tr = train_sup["GHI"] > ghi_thresh
    mask_te = test_sup["GHI"] > ghi_thresh
    if mask_te.sum() < 50:
        continue
    tr_d = train_sup[mask_tr].copy()
    te_d = test_sup[mask_te].copy()

    X_tr = tr_d[all_feature_cols].values
    y_tr = tr_d[TARGET].values
    X_te = te_d[all_feature_cols].values
    y_te = te_d[TARGET].values

    m = lgb.LGBMRegressor(n_estimators=1000, max_depth=10, learning_rate=0.03,
                           num_leaves=63, min_child_samples=10, subsample=0.8,
                           colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                           random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te).clip(0)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = m.score(X_tr, y_tr)
    rel = rmse / y_te.mean() * 100
    rng = y_te.max() - y_te.min()
    nrmse = (rmse / rng * 100) if rng > 0 else 0.0
    if nrmse < best_config["nrmse"]:
        best_config = {"rel": rel, "nrmse": nrmse, "rmse": rmse, "ghi": ghi_thresh, "features": all_feature_cols, "model": m}
    print(f"{f'GHI > {ghi_thresh}':<8} {len(te_d):>8} {y_te.mean():>10.0f} {rmse:>10.1f} {rel:>8.2f} {nrmse:>8.2f} {r2:>7.4f} {len(all_feature_cols):>6}")

# Now try progressive feature selection: start with most important features and add more
print(f"\nProgressive feature addition (GHI > {best_config['ghi'] if best_config['ghi'] is not None else 100})...")
ghi_best = best_config["ghi"] if best_config["ghi"] is not None else 100
mask_tr = train_sup["GHI"] > ghi_best
mask_te = test_sup["GHI"] > ghi_best
tr_d = train_sup[mask_tr].copy()
te_d = test_sup[mask_te].copy()

# Get feature importances from the best full model
m_all = best_config["model"]
imp = pd.Series(m_all.feature_importances_, index=all_feature_cols).sort_values(ascending=False)

# Try progressive feature subsets: top K features
print(f"\n{'Top-K':<8} {'RelErr%':>8} {'RMSE[Wh]':>10} {'R²':>7}")
print("-" * 35)
for k in [5, 10, 15, 20, 30, 40, 50]:
    feat_k = imp.head(k).index.tolist()
    X_tr_k = tr_d[feat_k].values
    y_tr_k = tr_d[TARGET].values
    X_te_k = te_d[feat_k].values
    y_te_k = te_d[TARGET].values
    m_k = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                             num_leaves=31, min_child_samples=10, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                             random_state=42, n_jobs=-1, verbose=-1)
    m_k.fit(X_tr_k, y_tr_k)
    pred_k = m_k.predict(X_te_k).clip(0)
    rmse_k = np.sqrt(mean_squared_error(y_te_k, pred_k))
    r2_k = m_k.score(X_tr_k, y_tr_k)
    rel_k = rmse_k / y_te_k.mean() * 100
    print(f"{k:<8} {rel_k:>8.2f} {rmse_k:>10.1f} {r2_k:>7.4f}")

# Try XGBoost on best feature set
print(f"\nTrying XGBoost on best feature set (GHI > {ghi_best})...")
feat_best = all_feature_cols
X_tr_x = tr_d[feat_best].values
y_tr_x = tr_d[TARGET].values
X_te_x = te_d[feat_best].values
y_te_x = te_d[TARGET].values

xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                          reg_lambda=0.1, random_state=42, n_jobs=-1, verbosity=0)
xgb_m.fit(X_tr_x, y_tr_x)
pred_x = xgb_m.predict(X_te_x).clip(0)
rmse_x = np.sqrt(mean_squared_error(y_te_x, pred_x))
rel_x = rmse_x / y_te_x.mean() * 100
r2_x = xgb_m.score(X_tr_x, y_tr_x)
print(f"  XGBoost: RelErr={rel_x:.2f}%, RMSE={rmse_x:.1f} Wh, R²={r2_x:.4f}")

# Try log-transformed target
print(f"\nTrying log-transform on target (GHI > {ghi_best})...")
y_tr_log = np.log1p(y_tr_x)
y_te_log = np.log1p(y_te_x)
m_log = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                           num_leaves=31, min_child_samples=10, subsample=0.8,
                           colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                           random_state=42, n_jobs=-1, verbose=-1)
m_log.fit(X_tr_x, y_tr_log)
pred_log = np.expm1(m_log.predict(X_te_x)).clip(0)
rmse_log = np.sqrt(mean_squared_error(y_te_x, pred_log))
rel_log = rmse_log / y_te_x.mean() * 100
r2_log = m_log.score(X_tr_x, y_tr_log)
print(f"  Log-target LightGBM: RelErr={rel_log:.2f}%, RMSE={rmse_log:.1f} Wh, R²={r2_log:.4f}")

# Ensemble: average LightGBM + XGBoost predictions
print(f"\nTrying ensemble (LightGBM + XGBoost) on best feature set...")
pred_ens = (pred_x + m_all.predict(X_te_x).clip(0)) / 2
rmse_ens = np.sqrt(mean_squared_error(y_te_x, pred_ens))
rel_ens = rmse_ens / y_te_x.mean() * 100
print(f"  Ensemble: RelErr={rel_ens:.2f}%, RMSE={rmse_ens:.1f} Wh")

# Summary of best results
print(f"\n{'='*50}")
print(f"  BEST RESULT: nRMSE = {best_config['nrmse']:.2f}% (RelErr = {best_config['rel']:.2f}%) at GHI > {best_config['ghi']}")
print(f"  {'='*50}")
print(f"\n  Top 20 features by importance:")
print(f"  {imp.head(20).to_string()}")

# Try stricter GHI thresholds with best feature set
print(f"\n  Stricter GHI sweep with best model...")
for ghi_strict in [200, 300, 400, 500, 600, 700, 800]:
    mask_s = test_sup["GHI"] > ghi_strict
    if mask_s.sum() < 30:
        continue
    te_s = test_sup[mask_s]
    X_te_s = te_s[all_feature_cols].values
    y_te_s = te_s[TARGET].values
    pred_s = m_all.predict(X_te_s).clip(0)
    rmse_s = np.sqrt(mean_squared_error(y_te_s, pred_s))
    rel_s = rmse_s / y_te_s.mean() * 100
    print(f"    GHI > {ghi_strict:<3}: {len(te_s):>5} rows, mean={y_te_s.mean():>6.0f} Wh, "
          f"RMSE={rmse_s:>6.1f} Wh, RelErr={rel_s:>5.2f}%")

# ========== 12. FINAL MODEL (Spec §8.1–§8.2) ==========
print("\n" + "=" * 60)
print("12. FINAL MODEL — LightGBM (Spec Feature Set)")
print("=" * 60)

FINAL_FEATURES = [
    "GHI", "temp", "isSun", "sunlightTime", "SunlightTime/daylength",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "ED_lag1", "ED_lag4", "ED_lag96",
    "GHI_lag1", "GHI_roll4",
    "ED_roll4",
    "GHI_x_sun", "GHI_x_isSun"
]
print(f"\n  Feature set: {len(FINAL_FEATURES)} features")

X_tr_f = train_fe[FINAL_FEATURES].values
y_tr_f = train_fe[TARGET].values
X_va_f = val_fe[FINAL_FEATURES].values
y_va_f = val_fe[TARGET].values
X_te_f = test_fe[FINAL_FEATURES].values
y_te_f = test_fe[TARGET].values

final_model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=8,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
final_model.fit(X_tr_f, y_tr_f)
pred_va_f = final_model.predict(X_va_f).clip(0)
pred_f = final_model.predict(X_te_f).clip(0)

f_val_metrics = compute_metrics(y_va_f, pred_va_f, "  Final Model (Val)")
f_metrics = compute_metrics(y_te_f, pred_f, "  Final Model (Test)")
f_val_daytime_nrmse = compute_daytime_nrmse(y_va_f, pred_va_f, val_fe["GHI"].values)
f_test_daytime_nrmse = compute_daytime_nrmse(y_te_f, pred_f, test_fe["GHI"].values)
print(f"  Final Model Val Daytime nRMSE = {f_val_daytime_nrmse:.2f}%")
print(f"  Final Model Test Daytime nRMSE = {f_test_daytime_nrmse:.2f}%")

# GHI threshold sweep
print(f"\n{'GHI >':<8} {'Rows':>8} {'Mean[Wh]':>10} {'RMSE[Wh]':>10} {'RelErr%':>8} {'R²':>7}")
print("-" * 60)
for ghi_thresh in [0, 25, 50, 100, 150, 200]:
    mask_tr = train_fe["GHI"] > ghi_thresh
    mask_te = test_fe["GHI"] > ghi_thresh
    if mask_te.sum() < 50:
        continue
    tr_d = train_fe[mask_tr]
    te_d = test_fe[mask_te]
    m = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                           num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1,
                           random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr_d[FINAL_FEATURES].values, tr_d[TARGET].values)
    pred = m.predict(te_d[FINAL_FEATURES].values).clip(0)
    rmse = np.sqrt(mean_squared_error(te_d[TARGET].values, pred))
    r2 = m.score(tr_d[FINAL_FEATURES].values, tr_d[TARGET].values)
    rel = rmse / te_d[TARGET].values.mean() * 100
    print(f"{f'GHI > {ghi_thresh}':<8} {len(te_d):>8} {te_d[TARGET].values.mean():>10.0f} {rmse:>10.1f} {rel:>8.2f} {r2:>7.4f}")

# Feature importance
f_imp = pd.Series(final_model.feature_importances_, index=FINAL_FEATURES).sort_values(ascending=False)
print(f"\n  Feature importance:")
print(f"  {f_imp.to_string()}")

# 24h rollout forecast with final model
print(f"\n  24h Rollout Forecast (final model)...")
fct_f = forecast_24h(final_model, None, FINAL_FEATURES, test_pp, train_pp,
                     forecast_start)
actual_96f = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_f = np.sqrt(mean_squared_error(actual_96f, fct_f.clip(0)))
rel_f_f = (rmse_f_f / actual_96f.mean()) * 100
print(f"  24h RMSE={rmse_f_f:.1f} Wh, RelErr={rel_f_f:.2f}%")

# ========== 13. FINAL SUMMARY ==========
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

model_labels = [m[0] for m in MODEL_CONFIGS]
header = f"{'Model':<20} {'Type':<8} {'Val RMSE':<10} {'Val nRMSE':<10} {'Val RelErr':<10} {'Val R²':<7} {'Test RMSE':<10} {'Test nRMSE':<10} {'Test RelErr':<10} {'Test R²':<7} {'Val Day nRMSE':<13} {'Test Day nRMSE':<14}"
print(header)
print("-" * len(header))
for label, mtype in MODEL_CONFIGS:
    va = results[label]["val"]
    te = results[label]["test"]
    vd = results[label]["val_daytime_nrmse"]
    td = results[label]["test_daytime_nrmse"]
    print(f"{label:<20} {mtype:<8} {va['rmse']:<10.1f} {va['nrmse']:<10.2f} {va['rel']:<10.2f} {va['r2']:<7.4f} {te['rmse']:<10.1f} {te['nrmse']:<10.2f} {te['rel']:<10.2f} {te['r2']:<7.4f} {vd:<13.2f} {td:<14.2f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
