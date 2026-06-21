"""
TRAINING — Train default models + hyperparameter tuning on validation
"""
import os, sys, warnings, random, pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import optuna

from config import TARGET, TOP5_FEATURES, FINAL_FEATURES, MODEL_CONFIGS

np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

FEATURE_SET = TOP5_FEATURES

print("=" * 60)
print("TRAINING — Load data, train models, hyperparameter tuning")
print("=" * 60)

train_fe = pd.read_pickle("data/train_fe.pkl")
val_fe = pd.read_pickle("data/val_fe.pkl")
test_fe = pd.read_pickle("data/test_fe.pkl")
train_pp = pd.read_pickle("data/train_pp.pkl")
val_pp = pd.read_pickle("data/val_pp.pkl")
test_pp = pd.read_pickle("data/test_pp.pkl")

print(f"Train FE: {len(train_fe):,} rows")
print(f"Val FE:   {len(val_fe):,} rows")
print(f"Test FE:  {len(test_fe):,} rows")

def compute_metrics(y_true, y_pred, label=""):
    y_pred = y_pred.clip(0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rel = (rmse / y_true.mean()) * 100
    rng = y_true.max() - y_true.min()
    nrmse = (rmse / rng * 100) if rng > 0 else 0.0
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"{label}")
    print(f"  MSE  = {mse:>12.1f} Wh\u00b2")
    print(f"  MAE  = {mae:>12.1f} Wh")
    print(f"  RMSE = {rmse:>12.1f} Wh")
    print(f"  RelErr = {rel:>8.2f} %")
    print(f"  nRMSE  = {nrmse:>8.2f} %")
    print(f"  R\u00b2   = {r2:>12.4f}")
    return dict(mse=mse, mae=mae, rmse=rmse, rel=rel, nrmse=nrmse, r2=r2)

def compute_daytime_nrmse(y_true, y_pred, ghi_values):
    daytime_mask = ghi_values > 0
    y_d = y_true[daytime_mask]
    yp_d = y_pred[daytime_mask].clip(0)
    rmse_d = np.sqrt(mean_squared_error(y_d, yp_d))
    rng_d = y_d.max() - y_d.min()
    return (rmse_d / rng_d * 100) if rng_d > 0 else 0.0

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
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, min_child_weight=5,
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
    print(f"{label} Training R\u00b2: {train_r2:.4f}")
    return model, scaler

models = {}
scalers = {}
results = {}

for label, mtype in MODEL_CONFIGS:
    print(f"\n--- {label} ---")
    model, scaler = train_model(train_fe, FEATURE_SET, mtype, label)
    models[label] = model
    scalers[label] = scaler
    X_tr = train_fe[FEATURE_SET].values
    if scaler is not None:
        X_tr = scaler.transform(X_tr)
    y_train_pred = model.predict(X_tr).clip(0)
    train_metrics = compute_metrics(train_fe[TARGET].values, y_train_pred, f"  {label} (Train)")
    X_va = val_fe[FEATURE_SET].values
    if scaler is not None:
        X_va = scaler.transform(X_va)
    y_val_pred = model.predict(X_va).clip(0)
    val_metrics = compute_metrics(val_fe[TARGET].values, y_val_pred, f"  {label} (Val)")
    val_daytime_nrmse = compute_daytime_nrmse(val_fe[TARGET].values, y_val_pred, val_fe["GHI"].values)
    print(f"  {label} Val Daytime nRMSE = {val_daytime_nrmse:.2f}%")
    results[label] = {"train": train_metrics, "val": val_metrics,
                       "val_daytime_nrmse": val_daytime_nrmse}

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (baseline)")
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
        # Lấy mẫu NGẪU NHIÊN (không phải 5000 dòng đầu theo thời gian) để đại diện cho toàn bộ phân phối train, tránh thiên lệch theo mùa
        sample_size = min(5000, len(train_fe))
        sample_idx = np.random.RandomState(42).choice(len(train_fe), size=sample_size, replace=False)
        X_tmp = train_fe[feature_list].values[sample_idx]
        y_tmp = train_fe[target_col].values[sample_idx]
        if scaler is not None:
            X_tmp = scaler.transform(X_tmp)
        r = permutation_importance(model, X_tmp, y_tmp, n_repeats=5, random_state=42, n_jobs=-1)
        return pd.Series(r.importances_mean, index=feature_list).sort_values(ascending=False)

importance_dfs = {}
for label, mtype in MODEL_CONFIGS:
    imp = compute_importance(models[label], scalers[label], FEATURE_SET, train_fe, mtype)
    importance_dfs[label] = imp

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (Optuna)")
print("=" * 60)

X_tr_all = train_fe[FEATURE_SET].values
y_tr_all = train_fe[TARGET].values

print("\n--- Tuning LightGBM with Optuna ---")
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    rmse_scores = []
    for tr_idx, va_idx in tscv.split(X_tr_all):
        X_fold_tr, X_fold_va = X_tr_all[tr_idx], X_tr_all[va_idx]
        y_fold_tr, y_fold_va = y_tr_all[tr_idx], y_tr_all[va_idx]
        model.fit(X_fold_tr, y_fold_tr)
        preds = model.predict(X_fold_va).clip(0)
        rmse_scores.append(np.sqrt(mean_squared_error(y_fold_va, preds)))
    return np.mean(rmse_scores)

lgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
print(f"\n  Best trial: {lgb_study.best_trial.number}")
print(f"  Best value (RMSE): {lgb_study.best_value:.1f} Wh")
print(f"  Best params: {lgb_study.best_params}")

with open("data/best_params.json", "w") as f:
    json.dump(lgb_study.best_params, f, indent=2)
    print(f"  Saved best params to data/best_params.json")

X_tr_fin = train_fe[FINAL_FEATURES].values
y_tr_fin = train_fe[TARGET].values

print("\n--- LightGBM with FINAL_FEATURES (default params) ---")
lgb_fin_default = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
lgb_fin_default.fit(X_tr_fin, y_tr_fin)
models["LightGBM (final)"] = lgb_fin_default
scalers["LightGBM (final)"] = None
y_tr_pred_fin = lgb_fin_default.predict(X_tr_fin).clip(0)
tr_m_fin = compute_metrics(train_fe[TARGET].values, y_tr_pred_fin, "  LightGBM (final) (Train)")
X_va_fin = val_fe[FINAL_FEATURES].values
y_va_pred_fin = lgb_fin_default.predict(X_va_fin).clip(0)
va_m_fin = compute_metrics(val_fe[TARGET].values, y_va_pred_fin, "  LightGBM (final) (Val)")
va_dnrmse_fin = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred_fin, val_fe["GHI"].values)
print(f"  LightGBM (final) Val Daytime nRMSE = {va_dnrmse_fin:.2f}%")
X_te_fin = test_fe[FINAL_FEATURES].values
y_te_pred_fin = lgb_fin_default.predict(X_te_fin).clip(0)
te_m_fin = compute_metrics(test_fe[TARGET].values, y_te_pred_fin, "  LightGBM (final) (Test)")
te_dnrmse_fin = compute_daytime_nrmse(test_fe[TARGET].values, y_te_pred_fin, test_fe["GHI"].values)
print(f"  LightGBM (final) Test Daytime nRMSE = {te_dnrmse_fin:.2f}%")
results["LightGBM (final)"] = {"train": tr_m_fin, "val": va_m_fin, "test": te_m_fin,
                                "val_daytime_nrmse": va_dnrmse_fin, "test_daytime_nrmse": te_dnrmse_fin}
importance_dfs["LightGBM (final)"] = compute_importance(lgb_fin_default, None, FINAL_FEATURES, train_fe, "lightgbm")

print("\n--- Tuning LightGBM for FINAL_FEATURES ---")

def lgb_objective_final(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    rmse_scores = []
    for tr_idx, va_idx in tscv.split(X_tr_fin):
        X_fold_tr, X_fold_va = X_tr_fin[tr_idx], X_tr_fin[va_idx]
        y_fold_tr, y_fold_va = y_tr_fin[tr_idx], y_tr_fin[va_idx]
        model.fit(X_fold_tr, y_fold_tr)
        preds = model.predict(X_fold_va).clip(0)
        rmse_scores.append(np.sqrt(mean_squared_error(y_fold_va, preds)))
    return np.mean(rmse_scores)

lgb_study_fin = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
lgb_study_fin.optimize(lgb_objective_final, n_trials=30, show_progress_bar=True)
print(f"\n  Best trial: {lgb_study_fin.best_trial.number}")
print(f"  Best value (RMSE): {lgb_study_fin.best_value:.1f} Wh")
print(f"  Best params: {lgb_study_fin.best_params}")

with open("data/best_params_final.json", "w") as f:
    json.dump(lgb_study_fin.best_params, f, indent=2)
    print(f"  Saved best params to data/best_params_final.json")

print("\n--- Retraining LightGBM (tuned on TOP5) ---")
best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1, verbose=-1)
best_lgb.fit(X_tr_all, y_tr_all)
models["LightGBM (tuned)"] = best_lgb
scalers["LightGBM (tuned)"] = None

y_tr_pred = best_lgb.predict(X_tr_all).clip(0)
tr_m = compute_metrics(train_fe[TARGET].values, y_tr_pred, "  LightGBM (tuned) (Train)")
X_va_all = val_fe[FEATURE_SET].values
y_va_pred = best_lgb.predict(X_va_all).clip(0)
va_m = compute_metrics(val_fe[TARGET].values, y_va_pred, "  LightGBM (tuned) (Val)")
va_dnrmse = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred, val_fe["GHI"].values)
print(f"  LightGBM (tuned) Val Daytime nRMSE = {va_dnrmse:.2f}%")
results["LightGBM (tuned)"] = {"train": tr_m, "val": va_m,
                                "val_daytime_nrmse": va_dnrmse}
importance_dfs["LightGBM (tuned)"] = compute_importance(best_lgb, None, FEATURE_SET, train_fe, "lightgbm")

print("\n--- Retraining LightGBM (final, tuned on FINAL_FEATURES) ---")
best_lgb_fin = lgb.LGBMRegressor(**lgb_study_fin.best_params, random_state=42, n_jobs=-1, verbose=-1)
best_lgb_fin.fit(X_tr_fin, y_tr_fin)
models["LightGBM (final tuned)"] = best_lgb_fin
scalers["LightGBM (final tuned)"] = None

y_tr_pred_ft = best_lgb_fin.predict(X_tr_fin).clip(0)
tr_m_ft = compute_metrics(train_fe[TARGET].values, y_tr_pred_ft, "  LightGBM (final tuned) (Train)")
X_va_ft = val_fe[FINAL_FEATURES].values
y_va_pred_ft = best_lgb_fin.predict(X_va_ft).clip(0)
va_m_ft = compute_metrics(val_fe[TARGET].values, y_va_pred_ft, "  LightGBM (final tuned) (Val)")
va_dnrmse_ft = compute_daytime_nrmse(val_fe[TARGET].values, y_va_pred_ft, val_fe["GHI"].values)
print(f"  LightGBM (final tuned) Val Daytime nRMSE = {va_dnrmse_ft:.2f}%")
X_te_ft = test_fe[FINAL_FEATURES].values
y_te_pred_ft = best_lgb_fin.predict(X_te_ft).clip(0)
te_m_ft = compute_metrics(test_fe[TARGET].values, y_te_pred_ft, "  LightGBM (final tuned) (Test)")
te_dnrmse_ft = compute_daytime_nrmse(test_fe[TARGET].values, y_te_pred_ft, test_fe["GHI"].values)
print(f"  LightGBM (final tuned) Test Daytime nRMSE = {te_dnrmse_ft:.2f}%")
results["LightGBM (final tuned)"] = {"train": tr_m_ft, "val": va_m_ft, "test": te_m_ft,
                                      "val_daytime_nrmse": va_dnrmse_ft, "test_daytime_nrmse": te_dnrmse_ft}
importance_dfs["LightGBM (final tuned)"] = compute_importance(best_lgb_fin, None, FINAL_FEATURES, train_fe, "lightgbm")

full_model_configs = MODEL_CONFIGS + [("LightGBM (tuned)", "lightgbm_tuned"),
                                       ("LightGBM (final)", "lightgbm_default"),
                                       ("LightGBM (final tuned)", "lightgbm_final_tuned")]

for label, mtype in full_model_configs:
    print(f"\n{label} - Feature Importance:")
    print(importance_dfs[label].to_string())

def compute_vif(train_fe, feature_list):
    X_vif = train_fe[feature_list].values
    vifs = []
    for i, feat in enumerate(feature_list):
        X_others = np.delete(X_vif, i, axis=1)
        r2 = LinearRegression().fit(X_others, X_vif[:, i]).score(X_others, X_vif[:, i])
        vif = 1 / (1 - r2) if r2 < 1 else 999
        vifs.append((feat, vif))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

vif_df = compute_vif(train_fe, FEATURE_SET)
print("\nVIF Analysis:")
print(vif_df.to_string())

print("\n" + "=" * 60)
print("CORRELATION MATRIX (15 weather columns from Renewable.csv)")
print("=" * 60)

WEATHER_COLS = [
    "GHI", "temp", "pressure", "humidity", "wind_speed",
    "rain_1h", "snow_1h", "clouds_all", "isSun", "sunlightTime",
    "dayLength", "SunlightTime/daylength", "weather_type", "hour", "month",
]
raw_df = pd.read_csv("data/filled_renewable.csv")
corr_df = raw_df[WEATHER_COLS].corr()
print(corr_df.to_string())

pairs = []
for i in range(len(WEATHER_COLS)):
    for j in range(i + 1, len(WEATHER_COLS)):
        pairs.append((corr_df.values[i, j], WEATHER_COLS[i], WEATHER_COLS[j]))
pairs.sort(key=lambda x: x[0], reverse=True)

print("\n--- POSITIVE CORRELATION ---")
for val, a, b in [p for p in pairs if p[0] > 0]:
    print(f"  {a:>28} vs {b:<28}  {val:+.4f}")

print("\n--- NEGATIVE CORRELATION ---")
for val, a, b in [p for p in pairs if p[0] < 0]:
    print(f"  {a:>28} vs {b:<28}  {val:+.4f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_df.values, cmap="RdYlBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(WEATHER_COLS)))
ax.set_yticks(range(len(WEATHER_COLS)))
ax.set_xticklabels(WEATHER_COLS, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(WEATHER_COLS, fontsize=8)
for i in range(len(WEATHER_COLS)):
    for j in range(len(WEATHER_COLS)):
        val = corr_df.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
fig.colorbar(im, ax=ax, shrink=0.75)
ax.set_title("Weather Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weather_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved weather_correlation.png")

print("\n" + "=" * 60)
print("SAVING MODELS AND RESULTS")
print("=" * 60)

model_data = {
    "models": {k: v for k, v in models.items()},
    "scalers": {k: v for k, v in scalers.items()},
    "importance_dfs": {k: v for k, v in importance_dfs.items()},
    "results": {k: v for k, v in results.items()},
    "full_model_configs": full_model_configs,
    "FEATURE_SET": FEATURE_SET,
    "FINAL_FEATURES": FINAL_FEATURES,
}
with open("data/models.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("Saved models to data/models.pkl")

test_data = {
    "train_fe": train_fe,
    "val_fe": val_fe,
    "test_fe": test_fe,
    "train_pp": train_pp,
    "val_pp": val_pp,
    "test_pp": test_pp,
}
with open("data/split_data.pkl", "wb") as f:
    pickle.dump(test_data, f)
print("Saved split data to data/split_data.pkl")

print("\nTRAINING COMPLETE\n")
