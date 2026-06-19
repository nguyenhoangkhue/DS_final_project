import os, warnings, random
import numpy as np

np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

TARGET = "Energy delta[Wh]"
TARGET_DIR = "outputs"

# ============================================================
# Feature group definitions (Issue 2 — separate nowcast vs forecast)
# ============================================================
KNOWN_FUTURE_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "isSun", "sunlightTime", "dayLength", "SunlightTime/daylength",
]

OBSERVED_WEATHER_FEATURES = [
    "GHI", "temp", "pressure", "humidity", "wind_speed",
    "rain_1h", "snow_1h", "clouds_all", "weather_type",
]

AUTOREGRESSIVE_FEATURES = [
    "ED_lag1", "ED_lag4", "ED_lag96", "ED_roll4",
    "GHI_lag1", "GHI_roll4",
]

NOWCAST_FEATURE_SET = KNOWN_FUTURE_FEATURES + OBSERVED_WEATHER_FEATURES + AUTOREGRESSIVE_FEATURES

FORECAST_FEATURE_SET = KNOWN_FUTURE_FEATURES + AUTOREGRESSIVE_FEATURES + ["GHI_x_sun", "GHI_x_isSun"]

TOP5_FEATURES = [
    "GHI", "isSun", "sunlightTime", "SunlightTime/daylength", "temp"
]

EXPANDED_FEATURES = [
    "GHI", "temp", "isSun", "sunlightTime", "SunlightTime/daylength",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "ED_lag1", "ED_lag4", "ED_lag96", "ED_roll4",
    "GHI_lag1", "GHI_roll4",
    "GHI_x_sun", "GHI_x_isSun",
]

FINAL_FEATURES = EXPANDED_FEATURES

MODEL_COLORS = {
    "Linear Regression": "#7F77DD",
    "Decision Tree":     "#EF9F27",
    "Random Forest":     "#1D9E75",
    "LightGBM":          "#E06B22",
    "XGBoost":           "#7B287D",
    "MLP":               "#E05252",
    "LightGBM (tuned)":  "#D4833A",
    "MLP (tuned)":       "#C04040",
    "Forecast LGB":      "#2E86AB",
}

MODEL_CONFIGS = [
    ("Linear Regression", "linear"),
    ("Decision Tree",     "tree"),
    ("Random Forest",     "forest"),
    ("LightGBM",          "lightgbm"),
    ("XGBoost",           "xgboost"),
    ("MLP",               "mlp"),
]

DAYTIME_FEATURES = EXPANDED_FEATURES + ["GHI_lag96", "ED_roll6h", "ED_roll24h", "GHI_roll24h"]

REF_RED = "#E05252"
