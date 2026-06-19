"""Debug 24h forecast predictions vs actuals"""
import pandas as pd, numpy as np

# Re-run the pipeline sections needed
exec(open("pipeline.py").read())

# Get first day of test set (instead of best day)
first_day_ts = test_fe.index[0]
fc_end_d = first_day_ts + pd.Timedelta(hours=23, minutes=45)

print(f"\nFirst day: {first_day_ts.date()}")
print(f"FEATURE_SET used: {FEATURE_SET}")

for label in ["Linear Regression", "LightGBM"]:
    fct = forecast_24h(
        models[label], scalers[label], FEATURE_SET,
        test_pp, train_pp, first_day_ts
    )
    actual = test_pp.loc[first_day_ts:fc_end_d, TARGET].values
    
    print(f"\n=== {label} (first day) ===")
    for i in range(0, 96, 12):
        ts = first_day_ts + pd.Timedelta(minutes=15*i)
        ghi = test_pp.loc[ts, "GHI"]
        print(f"  t={i:2d} {ts.time()}: pred={fct[i]:.1f}  actual={actual[i]:.1f}  GHI={ghi:.1f}")
    
    rmse = np.sqrt(np.mean((fct - actual)**2))
    print(f"  RMSE={rmse:.1f}  pred_mean={fct.mean():.1f}  actual_mean={actual.mean():.1f}")

# Check model coefficients for Linear Regression
lr = models["Linear Regression"]
print(f"\nLinear Regression coefficients:")
for feat, coef in zip(FEATURE_SET, lr.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  intercept: {lr.intercept_:.4f}")

# Test predict on a single midnight row (should be ~0)
print(f"\nSingle midnight prediction:")
midnight = test_pp.loc[first_day_ts]
X_mid = np.array([[midnight[f] for f in FEATURE_SET]])
print(f"  Features: {dict(zip(FEATURE_SET, X_mid[0]))}")
print(f"  Predicted: {lr.predict(X_mid)[0]:.1f}")

# Test predict on a noon row
noon_ts = first_day_ts + pd.Timedelta(hours=12)
noon = test_pp.loc[noon_ts]
X_noon = np.array([[noon[f] for f in FEATURE_SET]])
print(f"  Noon features: {dict(zip(FEATURE_SET, X_noon[0]))}")
print(f"  Noon predicted: {lr.predict(X_noon)[0]:.1f}  actual: {noon[TARGET]:.1f}")
