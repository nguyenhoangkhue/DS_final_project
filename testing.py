"""
TESTING — Evaluate models, visualizations, daytime model, final model, summary
"""
import os, sys, warnings, random, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

from config import (
    TARGET, TARGET_DIR, TOP5_FEATURES, EXPANDED_FEATURES,
    MODEL_COLORS, MODEL_CONFIGS, DAYTIME_FEATURES, REF_RED,
    OBSERVED_WEATHER_FEATURES, FORECAST_FEATURE_SET,
)

np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")
os.makedirs(TARGET_DIR, exist_ok=True)

# ============================================================
# 1. LOAD SAVED MODELS AND DATA
# ============================================================
print("=" * 60)
print("TESTING — Evaluation, experiments, visualizations")
print("=" * 60)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

models = mdata["models"]
scalers = mdata["scalers"]
importance_dfs = mdata["importance_dfs"]
full_model_configs = mdata["full_model_configs"]
FEATURE_SET = mdata["FEATURE_SET"]

with open("data/split_data.pkl", "rb") as f:
    sdata = pickle.load(f)

train_fe = sdata["train_fe"]
val_fe = sdata["val_fe"]
test_fe = sdata["test_fe"]
train_pp = sdata["train_pp"]
val_pp = sdata["val_pp"]
test_pp = sdata["test_pp"]

# ============================================================
# 1b. COMPUTE ALL RESULTS FROM SCRATCH (train/val/test)
# ============================================================
def compute_metrics(y_true, y_pred, label=""):
    y_pred = y_pred.clip(0)
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    rel = (rmse / y_true.mean()) * 100
    rng = y_true.max() - y_true.min()
    nrmse = (rmse / rng * 100) if rng > 0 else 0.0
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    if label:
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

FINAL_FEATURES = mdata.get("FINAL_FEATURES", EXPANDED_FEATURES)

results = {}
for label in models:
    feat_set = FINAL_FEATURES if "final" in label else FEATURE_SET
    X_tr = train_fe[feat_set].values
    if scalers[label] is not None:
        X_tr = scalers[label].transform(X_tr)
    y_tr_pred = models[label].predict(X_tr).clip(0)
    tr_m = compute_metrics(train_fe[TARGET].values, y_tr_pred)

    X_va = val_fe[feat_set].values
    if scalers[label] is not None:
        X_va = scalers[label].transform(X_va)
    y_va_pred = models[label].predict(X_va).clip(0)
    va_m = compute_metrics(val_fe[TARGET].values, y_va_pred)

    X_te = test_fe[feat_set].values
    if scalers[label] is not None:
        X_te = scalers[label].transform(X_te)
    y_te_pred = models[label].predict(X_te).clip(0)
    te_m = compute_metrics(test_fe[TARGET].values, y_te_pred)

    va_dnrmse = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred, val_fe["GHI"].values)
    te_dnrmse = compute_daytime_nrmse(test_fe[TARGET].values, y_te_pred, test_fe["GHI"].values)

    results[label] = {"train": tr_m, "val": va_m, "test": te_m,
                       "val_daytime_nrmse": va_dnrmse, "test_daytime_nrmse": te_dnrmse}

print(f"Loaded {len(full_model_configs)} models")
print(f"Test FE: {len(test_fe):,} rows")

# ============================================================
# 1c. BUILD CLIMATOLOGY (Issue 1 — replace future weather with historical averages)
# ============================================================
print("Building climatology for forecast features...")
climatology = {}
for col in OBSERVED_WEATHER_FEATURES:
    if col in train_pp.columns:
        climatology[col] = train_pp.groupby([train_pp.index.month, train_pp.index.hour])[col].mean()

# ============================================================
# 2. EDA
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
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
colors_bar = ["#E05252" if v < 0 else "#378ADD" for v in target_corr.values]
ax2.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor="white", linewidth=0.5)
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

# ============================================================
# 3. 24-HOUR ROLLOUT FORECAST (with climatology, no weather leakage)
# ============================================================
print("\n" + "=" * 60)
print("24-HOUR ROLLOUT FORECAST (climatology-based, no weather leakage)")
print("=" * 60)

def forecast_24h(model, scaler, feature_list, test_pp, train_pp,
                 forecast_start_time, target_col=TARGET, use_climatology=False):
    pre = test_pp.loc[:forecast_start_time - pd.Timedelta("15min")]
    full = pd.concat([train_pp[[target_col, "GHI"]], pre[[target_col, "GHI"]]])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    buf_ed = list(full[target_col].values[-200:])
    buf_ghi = list(full["GHI"].values[-200:])

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
                if use_climatology:
                    ghi_c = climatology["GHI"].get((idx.month, idx.hour), 0.0)
                else:
                    ghi_c = row["GHI"]
                feat[f] = ghi_c * row["SunlightTime/daylength"]
            elif f == "GHI_x_isSun":
                if use_climatology:
                    ghi_c = climatology["GHI"].get((idx.month, idx.hour), 0.0)
                else:
                    ghi_c = row["GHI"]
                feat[f] = ghi_c * row["isSun"]
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
            elif f in OBSERVED_WEATHER_FEATURES and use_climatology:
                feat[f] = climatology.get(f, pd.Series()).get((idx.month, idx.hour), 0.0)
            else:
                feat[f] = row[f]

        X = np.array([[feat[f] for f in feature_list]])
        if scaler is not None:
            X = scaler.transform(X)
        y_pred = float(max(0.0, model.predict(X)[0]))
        predictions.append(y_pred)

        buf_ed.append(y_pred)
        if use_climatology and "GHI" in climatology:
            buf_ghi.append(float(climatology["GHI"].get((idx.month, idx.hour), 0.0)))
        else:
            buf_ghi.append(float(row["GHI"]))

    return np.array(predictions)

def evaluate_by_horizon(model, scaler, feature_list, test_pp, train_pp, n_windows=30, use_climatology=False):
    errors_by_step = np.zeros((n_windows, 96))
    starts = pd.date_range(test_pp.index.min(), test_pp.index.max() - pd.Timedelta(hours=24),
                            periods=n_windows)
    valid_windows = 0
    for i, start in enumerate(starts):
        start_ts = test_pp.index[test_pp.index.searchsorted(start)]
        fct = forecast_24h(model, scaler, feature_list, test_pp, train_pp, start_ts, use_climatology=use_climatology)
        actual = test_pp.loc[start_ts:start_ts + pd.Timedelta(hours=23, minutes=45), TARGET].values
        if len(actual) == 96:
            errors_by_step[valid_windows] = (actual - fct) ** 2
            valid_windows += 1
    errors_by_step = errors_by_step[:valid_windows]
    rmse_by_step = np.sqrt(errors_by_step.mean(axis=0))
    overall_rmse = np.sqrt(errors_by_step.mean())
    return rmse_by_step, overall_rmse, valid_windows

# Issue 5: Don't cherry-pick best day — use a random representative day
rng = np.random.RandomState(42)
all_days = test_fe.index.normalize().unique()
test_daily_sum = test_fe[TARGET].resample("D").sum()
# pick a day with near-median total energy (representative, not best/worst)
median_sum = test_daily_sum.median()
best_day = test_daily_sum.sub(median_sum).abs().idxmin()
best_day_start_fe = int(test_fe.index.searchsorted(best_day))
forecast_start = test_fe.index[best_day_start_fe]
print(f"Representative forecast day: {forecast_start.date()} (median-energy day)")

def model_feature_set(label):
    return FINAL_FEATURES if "final" in label else FEATURE_SET

# --- Nowcast rollout (uses real weather — reveals weather leakage baseline) ---
print("\n--- 24h Rollout with NOWCAST (observed weather, reference) ---")
forecasts_nc = {}
for label, mtype in full_model_configs:
    fs = model_feature_set(label)
    fct = forecast_24h(models[label], scalers[label], fs, test_pp, train_pp,
                       forecast_start, use_climatology=False)
    forecasts_nc[label] = fct
    fc_end = forecast_start + pd.Timedelta(hours=23, minutes=45)
    actual = test_pp.loc[forecast_start:fc_end, TARGET].values
    rmse_f = np.sqrt(mean_squared_error(actual, fct.clip(0)))
    rng_f = actual.max() - actual.min()
    nrmse_f = (rmse_f / rng_f * 100) if rng_f > 0 else 0
    print(f"  {label} 24h RMSE={rmse_f:.1f} Wh, nRMSE={nrmse_f:.2f}%")

# Figure: Nowcast comparison — separate panel per model
time_axis = pd.date_range(forecast_start, forecast_start + pd.Timedelta(hours=23, minutes=45), periods=96)
n_models = len(full_model_configs)
ncols, nrows = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
axes = axes.flatten()
for idx, (label, mtype) in enumerate(full_model_configs):
    ax = axes[idx]
    fct = forecasts_nc[label].clip(0)
    _rng = actual.max() - actual.min()
    nrmse_val = (np.sqrt(mean_squared_error(actual, fct)) / _rng * 100) if _rng > 0 else 0
    ax.plot(time_axis, actual, color="black", linewidth=1.5, label="Actual")
    ax.plot(time_axis, fct, color=MODEL_COLORS.get(label, "#888888"), linewidth=1.2, alpha=0.85, label="Forecast")
    ax.set_title(f"{label} — nRMSE={nrmse_val:.2f}%", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if idx >= 6:
        ax.set_xlabel("Time", fontsize=9)
    if idx % 3 == 0:
        ax.set_ylabel(f"{TARGET} (Wh)", fontsize=9)
for idx in range(n_models, nrows * ncols):
    axes[idx].axis("off")
fig.suptitle(f"24-Hour Nowcast Rollout — {forecast_start.date()}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/13_nowcast_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 13_nowcast_comparison.png")

# Walk-forward evaluation on nowcast models
print("\n--- Walk-Forward Evaluation (nowcast, n=30 windows) ---")
for label, mtype in full_model_configs:
    fs = model_feature_set(label)
    rmse_by_step, overall_rmse, nw = evaluate_by_horizon(
        models[label], scalers[label], fs, test_pp, train_pp,
        n_windows=30, use_climatology=False)
    print(f"  {label:<22} avg 24h RMSE={overall_rmse:.1f} Wh ({nw} windows)")

# --- Forecast model rollout (uses climatology — honest forecast) ---
print("\n--- Training FORECAST model (no future weather) ---")
lgb_fc = lgb.LGBMRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03,
                            num_leaves=31, min_child_samples=10, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            random_state=42, n_jobs=-1, verbose=-1)
X_tr_fc = train_fe[FORECAST_FEATURE_SET].values
y_tr_fc = train_fe[TARGET].values
lgb_fc.fit(X_tr_fc, y_tr_fc)
print(f"  Training R²: {lgb_fc.score(X_tr_fc, y_tr_fc):.4f}")

# Single-day rollout with climatology (honest forecast)
fct_fc = forecast_24h(lgb_fc, None, FORECAST_FEATURE_SET, test_pp, train_pp,
                      forecast_start, use_climatology=True)
actual_fc = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_fc = np.sqrt(mean_squared_error(actual_fc, fct_fc.clip(0)))
rng_fc = actual_fc.max() - actual_fc.min()
nrmse_fc = (rmse_fc / rng_fc * 100) if rng_fc > 0 else 0
print(f"\n  Forecast Model 24h RMSE={rmse_fc:.1f} Wh, nRMSE={nrmse_fc:.2f}% (climatology)")

# Figure: True forecast comparison (climatology, no future weather)
fig, ax = plt.subplots(figsize=(14, 5))
time_axis_fc = pd.date_range(forecast_start, forecast_start + pd.Timedelta(hours=23, minutes=45), periods=96)
ax.plot(time_axis_fc, actual_fc, color="black", linewidth=2, label="Actual")
ax.plot(time_axis_fc, fct_fc.clip(0), color="#E05252", linewidth=1.5, alpha=0.85,
        label=f"Forecast LGB (climatology) — nRMSE={nrmse_fc:.2f}%")
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel(f"{TARGET} (Wh)", fontsize=12)
ax.set_title(f"True 24-Hour Forecast (no future weather) — {forecast_start.date()}", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/14_true_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 14_true_forecast.png")

# Walk-forward on forecast model (Issue 7)
print("\n--- Walk-Forward Evaluation (forecast model, n=50 windows) ---")
fc_rmse_by_step, fc_overall_rmse, fc_nw = evaluate_by_horizon(
    lgb_fc, None, FORECAST_FEATURE_SET, test_pp, train_pp,
    n_windows=50, use_climatology=True)
print(f"  Forecast Model avg 24h RMSE={fc_overall_rmse:.1f} Wh ({fc_nw} windows)")

# Issue 6: Error by horizon
steps_minutes = np.arange(1, 97) * 15
print(f"\n--- Error by Horizon (Forecast Model) ---")
for horizon_idx in [0, 23, 47, 71, 95]:
    print(f"  Step {horizon_idx+1:>2} ({steps_minutes[horizon_idx]:>4} min): RMSE={fc_rmse_by_step[horizon_idx]:.1f} Wh")

# Figure 10: Error by horizon
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps_minutes, fc_rmse_by_step, color="#2E86AB", linewidth=2)
ax.axhline(fc_overall_rmse, color=REF_RED, linestyle="--", alpha=0.7, label=f"Mean RMSE ({fc_overall_rmse:.1f} Wh)")
ax.set_xlabel("Forecast Horizon (minutes)", fontsize=12)
ax.set_ylabel("RMSE (Wh)", fontsize=12)
ax.set_title("Forecast Error by Horizon — Walk-Forward Across Test Set", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/10_error_by_horizon.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 10_error_by_horizon.png")

# ============================================================
# 4. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

# Figure 03: Model comparison table
table_data = []
for label, mtype in full_model_configs:
    tr = results[label]["train"]
    va = results[label]["val"]
    te = results[label]["test"]
    table_data.append([label, mtype,
                       f"{tr['rmse']:.1f}", f"{tr['rel']:.2f}%",
                       f"{va['rmse']:.1f}", f"{va['nrmse']:.2f}%", f"{va['rel']:.2f}%", f"{va['r2']:.4f}",
                       f"{te['rmse']:.1f}", f"{te['nrmse']:.2f}%", f"{te['rel']:.2f}%", f"{te['r2']:.4f}"])

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
model_labels = [m[0] for m in full_model_configs]

for label, mtype in full_model_configs:
    tr = results[label]["train"]
    va = results[label]["val"]
    te = results[label]["test"]
    for m in metric_names:
        key = {"RelErr": "rel", "nRMSE": "nrmse"}.get(m, m.lower())
        train_vals[m].append(tr[key])
        val_vals[m].append(va[key])
        test_vals[m].append(te[key])

n_mod = len(model_labels)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
x = np.arange(n_mod)
width = min(0.25, 0.7 / n_mod)
for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    ax.bar(x - width, train_vals[metric], width, label="Train",
           color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.35)
    ax.bar(x, val_vals[metric], width, label="Val",
           color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.7)
    ax.bar(x + width, test_vals[metric], width, label="Test",
           color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=1.0)
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
time_idx = pd.date_range(start=forecast_start, periods=96, freq="15min")
actual_vals = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values

n_models_f5 = len(full_model_configs)
n_cols_f5 = 2 if n_models_f5 > 3 else 1
n_rows_f5 = int(np.ceil(n_models_f5 / n_cols_f5))
fig, axes = plt.subplots(n_rows_f5, n_cols_f5, figsize=(7 * n_cols_f5, 4 * n_rows_f5), sharex=True)
axes = np.atleast_1d(axes).ravel()

for i, (label, mtype) in enumerate(full_model_configs):
    ax = axes[i]
    fct = forecasts_nc[label]
    rng_f5 = actual_vals.max() - actual_vals.min()
    nrmse_f5 = (np.sqrt(mean_squared_error(actual_vals, fct.clip(0))) / rng_f5 * 100) if rng_f5 > 0 else 0
    ax.plot(time_idx, actual_vals, color="black", linewidth=1.0, label="Actual")
    ax.plot(time_idx, fct, color=MODEL_COLORS[label], linewidth=1.0, linestyle="--", label="Predicted")
    ax.fill_between(time_idx, actual_vals, fct, alpha=0.15, color=MODEL_COLORS[label])
    ax.set_title(f"{label} — nRMSE: {nrmse_f5:.2f}%", fontsize=12, fontweight="bold")
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

# Figure 09: Full results dashboard
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 22))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
n09 = len(model_labels)
best_label = max(results, key=lambda k: results[k]["val"]["r2"])

ax1 = fig.add_subplot(gs[0, 0])
x09 = np.arange(n09)
w09 = min(0.25, 0.7 / n09)
train_rmses = [results[m[0]]["train"]["rmse"] for m in full_model_configs]
val_rmses = [results[m[0]]["val"]["rmse"] for m in full_model_configs]
test_rmses = [results[m[0]]["test"]["rmse"] for m in full_model_configs]
ax1.bar(x09 - w09, train_rmses, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.35)
ax1.bar(x09, val_rmses, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.7)
ax1.bar(x09 + w09, test_rmses, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs])
ax1.set_xticks(x09)
ax1.set_xticklabels(model_labels, fontsize=8)
ax1.set_title("RMSE Comparison", fontweight="bold")
ax1.set_ylabel("RMSE [Wh]")

ax2 = fig.add_subplot(gs[0, 1])
train_nrmse = [results[m[0]]["train"]["nrmse"] for m in full_model_configs]
val_nrmse = [results[m[0]]["val"]["nrmse"] for m in full_model_configs]
test_nrmse = [results[m[0]]["test"]["nrmse"] for m in full_model_configs]
ax2.bar(x09 - w09, train_nrmse, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.35)
ax2.bar(x09, val_nrmse, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.7)
ax2.bar(x09 + w09, test_nrmse, w09, color=[MODEL_COLORS[m[0]] for m in full_model_configs])
ax2.axhline(5, color=REF_RED, linestyle="--", linewidth=1.5)
ax2.set_xticks(x09)
ax2.set_xticklabels(model_labels, fontsize=8)
ax2.set_title("nRMSE Comparison", fontweight="bold")
ax2.set_ylabel("nRMSE [%]")

ax3 = fig.add_subplot(gs[1, :])
actual_96 = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
ax3.plot(time_idx, actual_96, color="black", linewidth=1.5, label="Actual")
for label, mtype in full_model_configs:
    if label in forecasts_nc:
        ax3.plot(time_idx, forecasts_nc[label], color=MODEL_COLORS[label], linewidth=1.0, linestyle="--", label=label)
ax3.set_title(f"24-Hour Forecast — {forecast_start.date()}", fontweight="bold")
ax3.set_ylabel("Energy delta [Wh]")
ax3.legend()
ax3.set_ylim(bottom=0)

n_steps = 96 * 14
time_2w = test_fe.index[:n_steps]

ax4 = fig.add_subplot(gs[2, 0])
actual_2w_best = test_fe[TARGET].iloc[:n_steps].values
X_2w_best = test_fe.iloc[:n_steps][model_feature_set(best_label)].values
if scalers[best_label] is not None:
    X_2w_best = scalers[best_label].transform(X_2w_best)
pred_2w_best = models[best_label].predict(X_2w_best).clip(0)
ax4.plot(time_2w, actual_2w_best, color="black", linewidth=0.4)
ax4.plot(time_2w, pred_2w_best, color=MODEL_COLORS[best_label], linewidth=0.5, linestyle="--")
ax4.set_title(f"{best_label} — 2-Week Prediction", fontweight="bold")
ax4.set_ylabel("Energy delta [Wh]")

ax5 = fig.add_subplot(gs[2, 1])
imp_best = importance_dfs[best_label]
colors_best = [MODEL_COLORS[best_label]] * len(imp_best)
ax5.barh(range(len(imp_best)), imp_best.values[::-1], color=colors_best[::-1], edgecolor="white")
ax5.set_yticks(range(len(imp_best)))
ax5.set_yticklabels(imp_best.index[::-1], fontsize=8)
ax5.set_title(f"{best_label} — Feature Importance", fontweight="bold")
best_mtype = dict(full_model_configs)[best_label]
xlabel_best = "|β_std|" if best_mtype == "linear" else "Importance"
ax5.set_xlabel(xlabel_best)
ax5.invert_yaxis()

ax6 = fig.add_subplot(gs[3, 0])
X_te_best6 = test_fe[model_feature_set(best_label)].values
if scalers[best_label] is not None:
    X_te_best6 = scalers[best_label].transform(X_te_best6)
y_pred_best6 = models[best_label].predict(X_te_best6).clip(0)
residuals_best6 = test_fe[TARGET].values - y_pred_best6
ax6.hist(residuals_best6, bins=80, color=MODEL_COLORS[best_label], edgecolor="white", alpha=0.8)
ax6.set_title(f"{best_label} — Residual Distribution", fontweight="bold")
ax6.set_xlabel("Residual [Wh]")
ax6.set_ylabel("Frequency")

ax7 = fig.add_subplot(gs[3, 1])
daytime_nrmses_val = [results[m[0]]["val_daytime_nrmse"] for m in full_model_configs]
daytime_nrmses_test = [results[m[0]]["test_daytime_nrmse"] for m in full_model_configs]
x_dn = np.arange(n09)
w_dn = min(0.35, 0.7 / n09)
ax7.bar(x_dn - w_dn/2, daytime_nrmses_val, w_dn,
        color=[MODEL_COLORS[m[0]] for m in full_model_configs], alpha=0.7)
ax7.bar(x_dn + w_dn/2, daytime_nrmses_test, w_dn,
        color=[MODEL_COLORS[m[0]] for m in full_model_configs])
ax7.set_xticks(x_dn)
ax7.set_xticklabels(model_labels, fontsize=7, rotation=20)
ax7.axhline(5, color=REF_RED, linestyle="--", linewidth=1.5)
ax7.set_title("Daytime-Only nRMSE (Val / Test)", fontweight="bold")
ax7.set_ylabel("nRMSE [%]")

fig.suptitle("Solar Energy Prediction — Full Pipeline Results\n"
             f"Train/Val/Test Split (70/15/15)  |  {n09} Model Types",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{TARGET_DIR}/09_full_results_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 09_full_results_dashboard.png")

# Figure 11: Time-domain comparison — LightGBM predictions vs actual across all splits
import matplotlib.dates as mdates
lgb_label = "LightGBM"
scaler_lgb = scalers.get(lgb_label)

# Compute predictions for all splits
def get_pred(split_df):
    X = split_df[FEATURE_SET].values
    if scaler_lgb is not None:
        X = scaler_lgb.transform(X)
    return models[lgb_label].predict(X).clip(0)

tr_pred = get_pred(train_fe)
va_pred = get_pred(val_fe)
te_pred = get_pred(test_fe)

# Predictions for tuned models on test set
te_pred_tuned = None
te_pred_final_tuned = None
if "LightGBM (tuned)" in models:
    X_te_fe = test_fe[FEATURE_SET].values
    s = scalers.get("LightGBM (tuned)")
    if s is not None:
        X_te_fe = s.transform(X_te_fe)
    te_pred_tuned = models["LightGBM (tuned)"].predict(X_te_fe).clip(0)
if "LightGBM (final tuned)" in models:
    X_te_f = test_fe[FINAL_FEATURES].values
    s = scalers.get("LightGBM (final tuned)")
    if s is not None:
        X_te_f = s.transform(X_te_f)
    te_pred_final_tuned = models["LightGBM (final tuned)"].predict(X_te_f).clip(0)

# Concatenate time indices and values across all splits
full_index = train_fe.index.append(val_fe.index).append(test_fe.index)
full_actual = np.concatenate([train_fe[TARGET].values, val_fe[TARGET].values, test_fe[TARGET].values])
full_pred = np.concatenate([tr_pred, va_pred, te_pred])
split_edges = [train_fe.index[-1], val_fe.index[-1]]

def plot_split_timeline(ax, time_idx, actual, pred, title, ylabel, show_legend):
    ax.plot(time_idx, actual, color="black", linewidth=0.4, alpha=0.7, label="Actual")
    ax.plot(time_idx, pred, color=MODEL_COLORS[lgb_label], linewidth=0.4, alpha=0.6, label=f"{lgb_label} Predicted")
    ax.fill_between(time_idx, actual, pred, alpha=0.12, color=MODEL_COLORS[lgb_label])
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Energy delta [Wh]")
    ax.set_xlabel("Time")
    if show_legend:
        ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Full timeline
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(full_index, full_actual, color="black", linewidth=0.3, alpha=0.7, label="Actual")
ax.plot(full_index, full_pred, color=MODEL_COLORS[lgb_label], linewidth=0.3, alpha=0.6, label=f"{lgb_label} Predicted")
for edge in split_edges:
    ax.axvline(x=edge, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.text(split_edges[0], ax.get_ylim()[1] * 0.97, "Train → Val", fontsize=8,
        ha="right", color="gray", style="italic")
ax.text(split_edges[1], ax.get_ylim()[1] * 0.97, "Val → Test", fontsize=8,
        ha="right", color="gray", style="italic")
ax.set_ylabel("Energy delta [Wh]")
ax.set_title("LightGBM — Full Timeline (Train / Val / Test)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/11_full_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 11_full_timeline.png")

# Train split
fig, ax = plt.subplots(figsize=(18, 5))
plot_split_timeline(ax, train_fe.index, train_fe[TARGET].values, tr_pred,
                    "LightGBM — Training Set", True, True)
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/11_train.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 11_train.png")

ONE_WEEK = 7 * 96
rng = np.random.RandomState(42)
train_start = rng.randint(0, max(1, len(train_fe) - ONE_WEEK))
train_slice = slice(train_start, train_start + ONE_WEEK)
fig, ax = plt.subplots(figsize=(18, 5))
plot_split_timeline(ax, train_fe.index[train_slice], train_fe[TARGET].values[train_slice], tr_pred[train_slice],
                    "LightGBM — Training Set (1-Week Zoom)", True, True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/11_train_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 11_train_zoom.png")

# Val split
fig, ax = plt.subplots(figsize=(18, 5))
plot_split_timeline(ax, val_fe.index, val_fe[TARGET].values, va_pred,
                    "LightGBM — Validation Set", True, True)
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/11_val.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 11_val.png")

val_start = rng.randint(0, max(1, len(val_fe) - ONE_WEEK))
val_slice = slice(val_start, val_start + ONE_WEEK)
fig, ax = plt.subplots(figsize=(18, 5))
plot_split_timeline(ax, val_fe.index[val_slice], val_fe[TARGET].values[val_slice], va_pred[val_slice],
                    "LightGBM — Validation Set (1-Week Zoom)", True, True)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.tight_layout()
plt.savefig(f"{TARGET_DIR}/11_val_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 11_val_zoom.png")

# Test split — separate images per model
LGB_LABELS = ["LightGBM", "LightGBM (tuned)", "LightGBM (final tuned)"]
LGB_PREDS  = [te_pred, te_pred_tuned, te_pred_final_tuned]
AVAIL = [(l, p) for l, p in zip(LGB_LABELS, LGB_PREDS) if p is not None]

test_start = rng.randint(0, max(1, len(test_fe) - ONE_WEEK))
test_slice = slice(test_start, test_start + ONE_WEEK)
t_idx = test_fe.index[test_slice]
t_act = test_fe[TARGET].values[test_slice]

for label, pred in AVAIL:
    safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    # Full test
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(test_fe.index, test_fe[TARGET].values, color="black", linewidth=0.4, alpha=0.7, label="Actual")
    ax.plot(test_fe.index, pred, color=MODEL_COLORS.get(label, "#888888"), linewidth=0.4, alpha=0.6, label="Predicted")
    ax.fill_between(test_fe.index, test_fe[TARGET].values, pred, alpha=0.12, color=MODEL_COLORS.get(label, "#888888"))
    ax.set_title(f"{label} — Test Set", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy delta [Wh]")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.tight_layout()
    plt.savefig(f"{TARGET_DIR}/11_test_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11_test_{safe}.png")

    # 1-week zoom
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(t_idx, t_act, color="black", linewidth=0.6, alpha=0.8, label="Actual")
    ax.plot(t_idx, pred[test_slice], color=MODEL_COLORS.get(label, "#888888"), linewidth=0.5, alpha=0.7, label="Predicted")
    ax.fill_between(t_idx, t_act, pred[test_slice], alpha=0.12, color=MODEL_COLORS.get(label, "#888888"))
    ax.set_title(f"{label} — Test Set (1-Week Zoom)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Energy delta [Wh]")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.tight_layout()
    plt.savefig(f"{TARGET_DIR}/11_test_{safe}_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 11_test_{safe}_zoom.png")

# ============================================================
# 5. TUNED TOP-5 EXPERIMENT (default vs tuned params, same feature set)
# ============================================================
print("\n" + "=" * 60)
print("TUNED TOP-5 EXPERIMENT (LightGBM)")
print("=" * 60)

with open("data/best_params.json", "r") as f:
    TUNED_TOP5_PARAMS = json.load(f)
print(f"  Loaded tuned params: {TUNED_TOP5_PARAMS}")

X_tr_t5 = train_fe[FEATURE_SET].values
y_tr_t5 = train_fe[TARGET].values
lgb_tuned_top5 = lgb.LGBMRegressor(**TUNED_TOP5_PARAMS, objective="tweedie", random_state=42, n_jobs=-1, verbose=-1)
lgb_tuned_top5.fit(X_tr_t5, y_tr_t5)
print(f"  Training R²: {lgb_tuned_top5.score(X_tr_t5, y_tr_t5):.4f}")

X_va_t5 = val_fe[FEATURE_SET].values
y_va_t5 = val_fe[TARGET].values
y_va_pred_t5 = lgb_tuned_top5.predict(X_va_t5).clip(0)
va_t5_metrics = compute_metrics(y_va_t5, y_va_pred_t5, "  Tuned Top-5 (Val)")

X_te_t5 = test_fe[FEATURE_SET].values
y_te_t5 = test_fe[TARGET].values
y_te_pred_t5 = lgb_tuned_top5.predict(X_te_t5).clip(0)
te_t5_metrics = compute_metrics(y_te_t5, y_te_pred_t5, "  Tuned Top-5 (Test)")

t5_val_daytime_nrmse = compute_daytime_nrmse(y_va_t5, y_va_pred_t5, val_fe["GHI"].values)
t5_test_daytime_nrmse = compute_daytime_nrmse(y_te_t5, y_te_pred_t5, test_fe["GHI"].values)
print(f"  Tuned Top-5 Val Daytime nRMSE = {t5_val_daytime_nrmse:.2f}%")
print(f"  Tuned Top-5 Test Daytime nRMSE = {t5_test_daytime_nrmse:.2f}%")

# 24h rollout forecast (tuned top-5)
print("\n  24h Rollout Forecast (tuned top-5)...")
fct_t5 = forecast_24h(lgb_tuned_top5, None, FEATURE_SET, test_pp, train_pp,
                      forecast_start)
actual_96_t5 = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_t5 = np.sqrt(mean_squared_error(actual_96_t5, fct_t5.clip(0)))
rng_f_t5 = actual_96_t5.max() - actual_96_t5.min()
nrmse_f_t5 = (rmse_f_t5 / rng_f_t5 * 100) if rng_f_t5 > 0 else 0
print(f"  24h RMSE={rmse_f_t5:.1f} Wh, nRMSE={nrmse_f_t5:.2f}%")

# 24h on val
val_fc_start = val_fe.index[0]
fct_t5_val = forecast_24h(lgb_tuned_top5, None, FEATURE_SET, val_pp, train_pp, val_fc_start)
actual_96_val_t5 = val_pp.loc[val_fc_start:val_fc_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_t5_val = np.sqrt(mean_squared_error(actual_96_val_t5, fct_t5_val.clip(0)))

baseline_label = "LightGBM"
bl_te = results[baseline_label]["test"]
bl_va = results[baseline_label]["val"]
print(f"\n{'Metric':<26} {'Default Top-5 Val':<18} {'Tuned Top-5 Val':<18} {'Default Top-5 Test':<18} {'Tuned Top-5 Test':<18}")
print("-" * 98)
for name, vva, eva, tva, eta in [
    ("Val RMSE [Wh]", f"{bl_va['rmse']:.1f}", f"{va_t5_metrics['rmse']:.1f}", "", ""),
    ("Test RMSE [Wh]", "", "", f"{bl_te['rmse']:.1f}", f"{te_t5_metrics['rmse']:.1f}"),
    ("Val RelErr [%]", f"{bl_va['rel']:.2f}", f"{va_t5_metrics['rel']:.2f}", "", ""),
    ("Test RelErr [%]", "", "", f"{bl_te['rel']:.2f}", f"{te_t5_metrics['rel']:.2f}"),
    ("Val nRMSE [%]", f"{bl_va['nrmse']:.2f}", f"{va_t5_metrics['nrmse']:.2f}", "", ""),
    ("Test nRMSE [%]", "", "", f"{bl_te['nrmse']:.2f}", f"{te_t5_metrics['nrmse']:.2f}"),
    ("Val Daytime nRMSE [%]", f"{results[baseline_label]['val_daytime_nrmse']:.2f}",
     f"{t5_val_daytime_nrmse:.2f}", "", ""),
    ("Test Daytime nRMSE [%]", "", "", f"{results[baseline_label]['test_daytime_nrmse']:.2f}",
     f"{t5_test_daytime_nrmse:.2f}"),
    ("Val R²", f"{bl_va['r2']:.4f}", f"{va_t5_metrics['r2']:.4f}", "", ""),
    ("Test R²", "", "", f"{bl_te['r2']:.4f}", f"{te_t5_metrics['r2']:.4f}"),
]:
    print(f"{name:<26} {vva:<18} {eva:<18} {tva:<18} {eta:<18}")

baseline_fct = forecasts_nc.get(baseline_label)
if baseline_fct is not None:
    baseline_rmse_t5 = np.sqrt(mean_squared_error(actual_96_t5, baseline_fct.clip(0)))
    baseline_nrmse_t5 = (baseline_rmse_t5 / (actual_96_t5.max() - actual_96_t5.min()) * 100) if actual_96_t5.max() > actual_96_t5.min() else 0
else:
    baseline_nrmse_t5 = 0
rng_t5_val = actual_96_val_t5.max() - actual_96_val_t5.min()
nrmse_f_t5_val = (rmse_f_t5_val / rng_t5_val * 100) if rng_t5_val > 0 else 0
print(f"{'24h Forecast nRMSE (Test)':<26} {baseline_nrmse_t5:<18.2f} {nrmse_f_t5:<18.2f}")
print(f"{'24h Forecast nRMSE (Val)':<26} {'':<18} {nrmse_f_t5_val:<18.2f}")

t5_imp = pd.Series(lgb_tuned_top5.feature_importances_, index=FEATURE_SET).sort_values(ascending=False)
print(f"\nTop-5 feature importance (tuned):")
print(t5_imp.to_string())

# ============================================================
# 6. DAYTIME-ONLY MODEL (5% Target)
# ============================================================
print("\n" + "=" * 60)
print("DAYTIME-ONLY MODEL (GHI > 0) -- TARGETING <= 5% nRMSE")
print("=" * 60)

# Tính 4 feature này trên dữ liệu CHRONOLOGICAL ĐẦY ĐỦ (train_fe/val_fe/test_fe)
# TRƯỚC khi filter GHI>0 -- để shift(96) đúng là "24h trước" thật, không bị lệch theo mùa
EXTRA_COLS = ["GHI_lag96", "ED_roll6h", "ED_roll24h", "GHI_roll24h"]
for d_full in (train_fe, val_fe, test_fe):
    d_full["GHI_lag96"] = d_full["GHI"].shift(96)
    d_full["ED_roll6h"] = d_full[TARGET].shift(1).rolling(window=24, min_periods=1).mean()
    d_full["ED_roll24h"] = d_full[TARGET].shift(1).rolling(window=96, min_periods=1).mean()
    d_full["GHI_roll24h"] = d_full["GHI"].shift(1).rolling(window=96, min_periods=1).mean()

train_day = train_fe[train_fe["GHI"] > 0].dropna(subset=EXTRA_COLS).copy()
val_day = val_fe[val_fe["GHI"] > 0].dropna(subset=EXTRA_COLS).copy()
test_day = test_fe[test_fe["GHI"] > 0].dropna(subset=EXTRA_COLS).copy()
print(f"  Daytime train: {len(train_day):,} rows ({len(train_day)/len(train_fe):.0%})")
print(f"  Daytime val:   {len(val_day):,} rows ({len(val_day)/len(val_fe):.0%})")
print(f"  Daytime test:  {len(test_day):,} rows ({len(test_day)/len(test_fe):.0%})")

DAYTIME_FEATURES_LOCAL = EXPANDED_FEATURES + EXTRA_COLS
print(f"  Using {len(DAYTIME_FEATURES_LOCAL)} features")

X_day = train_day[DAYTIME_FEATURES_LOCAL].values
y_day = train_day[TARGET].values

print("\n  Training daytime-only LightGBM (fixed params)...")
lgb_day = lgb.LGBMRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03,
                             num_leaves=63, min_child_samples=10, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                             objective="tweedie", tweedie_variance_power=1.3,
                             random_state=42, n_jobs=-1, verbose=-1)
lgb_day.fit(X_day, y_day)
print(f"  Training R²: {lgb_day.score(X_day, y_day):.4f}")

# Val
X_va_day = val_day[DAYTIME_FEATURES_LOCAL].values
y_va_day = val_day[TARGET].values
y_va_day_pred = lgb_day.predict(X_va_day).clip(0)
day_val_metrics = compute_metrics(y_va_day, y_va_day_pred, "  Daytime-Only Model (Val)")

# Test
X_te_day = test_day[DAYTIME_FEATURES_LOCAL].values
y_te_day = test_day[TARGET].values
y_te_day_pred = lgb_day.predict(X_te_day).clip(0)
day_metrics = compute_metrics(y_te_day, y_te_day_pred, "  Daytime-Only Model (Test)")
day_nrmse = day_metrics["nrmse"]

# Compare with best baseline
best_val_daytime_nrmse = min(results[m[0]]["val_daytime_nrmse"] for m in full_model_configs)
best_full_val_rel = min(results[m[0]]["val"]["rel"] for m in full_model_configs)
best_full_val_nrmse = min(results[m[0]]["val"]["nrmse"] for m in full_model_configs)
print(f"\n  {'='*50}")
print(f"  TARGET CHECK: nRMSE = {day_nrmse:.2f}%")
print(f"  {'PASS <= 5%' if day_nrmse <= 5 else 'FAIL > 5%'}")
print(f"  {'='*50}")
print(f"\n  {'Metric':<25} {'Best Full (Val)':<18} {'Daytime-Only':<18} {'Improvement':<12}")
print(f"  {'-'*73}")
print(f"  {'Val RelErr [%]':<25} {best_full_val_rel:<18.2f} {day_val_metrics['rel']:<18.2f} {best_full_val_rel - day_val_metrics['rel']:<+12.2f}")
print(f"  {'Val nRMSE [%]':<25} {best_full_val_nrmse:<18.2f} {day_val_metrics['nrmse']:<18.2f} {best_full_val_nrmse - day_val_metrics['nrmse']:<+12.2f}")
print(f"  {'Val Daytime nRMSE [%]':<25} {best_val_daytime_nrmse:<18.2f} {day_val_metrics['nrmse']:<18.2f} {best_val_daytime_nrmse - day_val_metrics['nrmse']:<+12.2f}")

# Multi-threshold test
print(f"\n  Testing stricter GHI thresholds...")
for ghi_thresh in [25, 50, 100]:
    tr_d = train_fe[train_fe["GHI"] > ghi_thresh].dropna(subset=EXTRA_COLS).copy()
    val_d = val_fe[val_fe["GHI"] > ghi_thresh].dropna(subset=EXTRA_COLS).copy()
    te_d = test_fe[test_fe["GHI"] > ghi_thresh].dropna(subset=EXTRA_COLS).copy()
    if len(te_d) < 100:
        continue
    bp = dict(n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
              min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=0.1)
    m = lgb.LGBMRegressor(**bp, objective="tweedie", tweedie_variance_power=1.3,
                           random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr_d[DAYTIME_FEATURES_LOCAL].values, tr_d[TARGET].values)
    r2_tr = m.score(tr_d[DAYTIME_FEATURES_LOCAL].values, tr_d[TARGET].values)
    pred_va = m.predict(val_d[DAYTIME_FEATURES_LOCAL].values).clip(0) if len(val_d) > 0 else np.array([])
    rmse_va = np.sqrt(mean_squared_error(val_d[TARGET].values, pred_va)) if len(pred_va) > 0 else float("nan")
    pred_te = m.predict(te_d[DAYTIME_FEATURES_LOCAL].values).clip(0)
    rmse_te = np.sqrt(mean_squared_error(te_d[TARGET].values, pred_te))
    rng_te = te_d[TARGET].values.max() - te_d[TARGET].values.min()
    nrmse_te = (rmse_te / rng_te * 100) if rng_te > 0 else 0
    print(f"  GHI > {ghi_thresh:<3}: TrainR²={r2_tr:.4f}  ValRMSE={rmse_va:.0f} Wh  "
          f"TestRMSE={rmse_te:.0f} Wh  nRMSE={nrmse_te:.2f}%"
          f"{'  ⚠ overfit' if r2_tr > 0.98 and rmse_va > rmse_te * 1.3 else ''}")

# ============================================================
# 8. FINAL MODEL (Spec Feature Set, tuned for this feature set)
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL — LightGBM (Spec Feature Set, dedicated tuning)")
print("=" * 60)

with open("data/best_params_final.json", "r") as f:
    FINAL_PARAMS = json.load(f)
print(f"  Loaded FINAL_FEATURES params: {FINAL_PARAMS}")
print(f"\n  Feature set: {len(FINAL_FEATURES)} features")
X_tr_f = train_fe[FINAL_FEATURES].values
y_tr_f = train_fe[TARGET].values
X_va_f = val_fe[FINAL_FEATURES].values
y_va_f = val_fe[TARGET].values
X_te_f = test_fe[FINAL_FEATURES].values
y_te_f = test_fe[TARGET].values

final_model = lgb.LGBMRegressor(**FINAL_PARAMS, objective="tweedie", random_state=42, n_jobs=-1, verbose=-1)
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
print(f"\n{'GHI >':<8} {'Rows':>8} {'Mean[Wh]':>10} {'RMSE[Wh]':>10} {'nRMSE%':>8} {'R²':>7}")
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
                           objective="tweedie", tweedie_variance_power=1.3,
                           random_state=42, n_jobs=-1, verbose=-1)
    va_d = val_fe[val_fe["GHI"] > ghi_thresh]
    m.fit(tr_d[FINAL_FEATURES].values, tr_d[TARGET].values)
    r2_tr    = m.score(tr_d[FINAL_FEATURES].values, tr_d[TARGET].values)
    pred_va  = m.predict(va_d[FINAL_FEATURES].values).clip(0) if len(va_d) > 0 else np.array([])
    rmse_va  = np.sqrt(mean_squared_error(va_d[TARGET].values, pred_va)) if len(pred_va) > 0 else float("nan")
    pred_te  = m.predict(te_d[FINAL_FEATURES].values).clip(0)
    rmse_te  = np.sqrt(mean_squared_error(te_d[TARGET].values, pred_te))
    rng_te = te_d[TARGET].values.max() - te_d[TARGET].values.min()
    nrmse_te = (rmse_te / rng_te * 100) if rng_te > 0 else 0
    print(f"{f'GHI > {ghi_thresh}':<8} {len(te_d):>8} {te_d[TARGET].values.mean():>10.0f} "
          f"TrainR²={r2_tr:.4f}  ValRMSE={rmse_va:>7.1f}  TestRMSE={rmse_te:>7.1f}  nRMSE={nrmse_te:.2f}%"
          f"{'  ⚠ overfit' if r2_tr > 0.98 and not np.isnan(rmse_va) and rmse_va > rmse_te * 1.3 else ''}")

# Feature importance
f_imp = pd.Series(final_model.feature_importances_, index=FINAL_FEATURES).sort_values(ascending=False)
print(f"\n  Feature importance:")
print(f"  {f_imp.to_string()}")

# 24h rollout
print(f"\n  24h Rollout Forecast (final model)...")
fct_f = forecast_24h(final_model, None, FINAL_FEATURES, test_pp, train_pp,
                     forecast_start, use_climatology=True)
actual_96f = test_pp.loc[forecast_start:forecast_start + pd.Timedelta(hours=23, minutes=45), TARGET].values
rmse_f_f = np.sqrt(mean_squared_error(actual_96f, fct_f.clip(0)))
rng_f_f = actual_96f.max() - actual_96f.min()
nrmse_f_f = (rmse_f_f / rng_f_f * 100) if rng_f_f > 0 else 0
print(f"  24h RMSE={rmse_f_f:.1f} Wh, nRMSE={nrmse_f_f:.2f}%")

# ============================================================
# 8. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

header = (f"{'Model':<20} {'Type':<8} {'Val RMSE':<10} {'Val nRMSE':<10} {'Val RelErr':<10} {'Val R²':<7} "
          f"{'Test RMSE':<10} {'Test nRMSE':<10} {'Test RelErr':<10} {'Test R²':<7} "
          f"{'Val Day nRMSE':<13} {'Test Day nRMSE':<14}")
print(header)
print("-" * len(header))
for label, mtype in full_model_configs:
    va = results[label]["val"]
    te = results[label]["test"]
    vd = results[label]["val_daytime_nrmse"]
    td = results[label]["test_daytime_nrmse"]
    print(f"{label:<20} {mtype:<8} {va['rmse']:<10.1f} {va['nrmse']:<10.2f} {va['rel']:<10.2f} {va['r2']:<7.4f} "
          f"{te['rmse']:<10.1f} {te['nrmse']:<10.2f} {te['rel']:<10.2f} {te['r2']:<7.4f} {vd:<13.2f} {td:<14.2f}")

# Forecast model summary (honest 24h forecast, no future weather)
fc_val_pred = lgb_fc.predict(val_fe[FORECAST_FEATURE_SET].values).clip(0)
fc_va = compute_metrics(val_fe[TARGET].values, fc_val_pred)
fc_te_pred = lgb_fc.predict(test_fe[FORECAST_FEATURE_SET].values).clip(0)
fc_te = compute_metrics(test_fe[TARGET].values, fc_te_pred)
fc_vd = compute_daytime_nrmse(val_fe[TARGET].values, fc_val_pred, val_fe["GHI"].values)
fc_td = compute_daytime_nrmse(test_fe[TARGET].values, fc_te_pred, test_fe["GHI"].values)
print(f"{'Forecast LGB (no wx)':<20} {'forecast':<8} {fc_va['rmse']:<10.1f} {fc_va['nrmse']:<10.2f} {fc_va['rel']:<10.2f} {fc_va['r2']:<7.4f} "
      f"{fc_te['rmse']:<10.1f} {fc_te['nrmse']:<10.2f} {fc_te['rel']:<10.2f} {fc_te['r2']:<7.4f} {fc_vd:<13.2f} {fc_td:<14.2f}")
print(f"\n  Walk-Forward Forecast (climatology, {fc_nw} windows): avg 24h RMSE = {fc_overall_rmse:.1f} Wh")

# ============================================================
# 9. EXPORT PREDICTIONS CSV
# ============================================================
Light_GBM_Tuned = "LightGBM (final tuned)"
if Light_GBM_Tuned in models:
    fs = FINAL_FEATURES
    X_te_exp = test_fe[fs].values
    s = scalers.get(Light_GBM_Tuned)
    if s is not None:
        X_te_exp = s.transform(X_te_exp)
    y_te_pred = models[Light_GBM_Tuned].predict(X_te_exp).clip(0)

    lgb_tuned_pred = None
    if "LightGBM (tuned)" in models:
        X_te_tuned = test_fe[FEATURE_SET].values
        s_t = scalers.get("LightGBM (tuned)")
        if s_t is not None:
            X_te_tuned = s_t.transform(X_te_tuned)
        lgb_tuned_pred = models["LightGBM (tuned)"].predict(X_te_tuned).clip(0)

    tuned_col = lgb_tuned_pred if lgb_tuned_pred is not None else np.full(len(test_fe), np.nan)
    out = pd.DataFrame({
        "timestamp": test_fe.index,
        "forecast_LGBM_final_tuned": y_te_pred.round(1),
        "forecast_LGBM_tuned": tuned_col.round(1),
        "EnergyDelta": test_fe[TARGET].values.round(1),
    })
    out.to_csv("data/predictions.csv", index=False)
    print("\nSaved data/predictions.csv")

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)