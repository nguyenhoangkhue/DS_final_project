"""
PREPROCESSING — Historical Pattern-Based Reconstruction + 70/15/15 split + Feature Engineering
"""
import pandas as pd
import numpy as np
from config import TARGET, TARGET_DIR

np.random.seed(42)

# ============================================================
# 1. LOAD
# ============================================================
print("=" * 60)
print("PREPROCESSING — LOAD, GAP-FILL FULL, SPLIT, ENGINEER")
print("=" * 60)

df = pd.read_csv("data/Renewable.csv")
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)
# ============================================================
# 1b. DETECT "SOFT" MISSING DATA: Energy=0 kéo dài bất thường khi GHI đủ cao
# Không phải NaN nên gap-detection hiện tại bỏ sót -- gắn lại thành NaN để xử lý chung
# ============================================================
GHI_ANOMALY_THRESHOLD = 40   # W/m² -- cao hơn hẳn dead-zone thật (median ~3.5, p75 ~9)
MIN_ANOMALY_RUN = 4           # >= 1 giờ liên tục (4 bước 15 phút)

is_anomalous_zero = (df[TARGET] == 0) & (df["GHI"] > GHI_ANOMALY_THRESHOLD)
run_id = is_anomalous_zero.ne(is_anomalous_zero.shift()).cumsum()
run_lengths = is_anomalous_zero.groupby(run_id).transform("sum")
flag_as_missing = is_anomalous_zero & (run_lengths >= MIN_ANOMALY_RUN)

n_flagged = flag_as_missing.sum()
if n_flagged > 0:
    print(f"Detected {n_flagged:,} anomalous Energy=0 rows "
          f"(GHI>{GHI_ANOMALY_THRESHOLD}, run >={MIN_ANOMALY_RUN} steps) "
          f"-- reassigned to NaN for gap-fill")
    df.loc[flag_as_missing, TARGET] = np.nan
# ============================================================
# 2a. HISTORICAL PATTERN-BASED RECONSTRUCTION (Gap Fill — chunking version)
# ============================================================
# Step 1   : Detect continuous missing gaps (from the target column)
# Step 1b  : Gap too close to the start of the dataset (not enough history even after
#            chunking) -> trim it (rare)
# Step 2   : Extract a 24h context right before each CHUNK (not the whole gap)
# Step 3   : Search historical windows of the same length as the CHUNK (not the whole gap)
# Step 4   : Copy the corresponding future sequence for ALL columns (except deterministic ones)
# Step 5   : Scale (additive for temp, multiplicative for the rest; skip rain/snow)
# ============================================================

def reindex_fill(df_in, target_col=TARGET, train_cutoff_time=None, val_cutoff_time=None,
                  chunk_size=96, seasonal_window=96):
    orig_dtypes = df_in.drop(columns=["Time"], errors="ignore").dtypes.to_dict()
    d = df_in.set_index("Time").sort_index()
    full_idx = pd.date_range(start=d.index.min(), end=d.index.max(), freq="15min", name="Time")
    d = d.reindex(full_idx)

    # Columns computed directly from the timestamp — NEVER go through pattern-matching
    DETERMINISTIC_COLS = {"hour", "month", "isSun", "sunlightTime", "dayLength", "SunlightTime/daylength"}
    d["hour"] = d.index.hour
    d["month"] = d.index.month

    # ============================================================
    # STEP 1: Detect continuous missing gaps
    # ============================================================
    is_missing = d[target_col].isna()
    gap_groups = is_missing.ne(is_missing.shift()).cumsum()
    gaps = []
    for gap_id in gap_groups[is_missing].unique():
        pos = np.where(gap_groups.values == gap_id)[0]
        gaps.append({"start": pos[0], "len": len(pos)})

    # ============================================================
    # STEP 1b: Final safety net — only triggers if a gap is so close
    # to the start of the dataset that even a single small chunk
    # doesn't have enough real history before it
    # (< chunk_size + seasonal_window rows).
    # ============================================================
    unreliable = [g for g in gaps if g["start"] < chunk_size + seasonal_window]
    trim_end_pos = 0
    if unreliable:
        worst = max(unreliable, key=lambda g: g["start"] + g["len"])
        trim_end_pos = worst["start"] + worst["len"]
        print(f"[reindex_fill] Trimming {trim_end_pos:,} leading rows: gap at position {worst['start']} "
              f"(length {worst['len']}) only has {worst['start']} rows of real history, "
              f"which is less than the {chunk_size + seasonal_window}-row minimum for one chunk.")
        d = d.iloc[trim_end_pos:]
        full_idx = d.index
        is_missing = d[target_col].isna()
        gap_groups = is_missing.ne(is_missing.shift()).cumsum()

    missing_mask = d[target_col].isna().copy()
    col_names = d.columns.tolist()
    arrays = {col: d[col].values.copy() for col in col_names}
    raw_arrays = {col: arr.copy() for col, arr in arrays.items()}   # IMMUTABLE — only for searching/scaling
    n_arr = len(arrays[target_col])

    # Block search from crossing the train -> val -> test boundary
    train_end_pos = full_idx.searchsorted(train_cutoff_time, side="right") if train_cutoff_time is not None else n_arr
    val_end_pos = full_idx.searchsorted(val_cutoff_time, side="right") if val_cutoff_time is not None else n_arr

    def split_of(pos):
        if pos < train_end_pos: return 0
        elif pos < val_end_pos: return 1
        return 2

    def window_allowed(start_pos, length, ref_split):
        end_pos = start_pos + length - 1
        return split_of(start_pos) <= ref_split and split_of(end_pos) <= ref_split

    def split_into_chunks(gap_start, gap_len, size):
        chunks, pos, remaining = [], gap_start, gap_len
        while remaining > 0:
            this_len = min(size, remaining)
            chunks.append((pos, this_len))
            pos += this_len
            remaining -= this_len
        return chunks

    for gap_id in gap_groups[is_missing].unique():
        gap_positions = np.where(gap_groups.values == gap_id)[0]
        gap_start, gap_len = gap_positions[0], len(gap_positions)
        if gap_start >= gap_len + seasonal_window:
            # Enough real history before the gap → search the whole gap as one block (original behaviour)
            chunks = [(gap_start, gap_len)]
        else:
            # Not enough history for a full-length search → chunk into seasonal_window-sized pieces
            chunks = split_into_chunks(gap_start, gap_len, chunk_size)

        for chunk_start, chunk_len in chunks:
            chunk_positions = np.arange(chunk_start, chunk_start + chunk_len)
            context_start = chunk_start - seasonal_window
            if context_start < 0:
                continue  # safety net — should almost never happen after Step 1b

            # STEP 2: Context — use "arrays" so it CHAINS from the previous chunk (if same gap)
            context = arrays[target_col][context_start:chunk_start]
            if np.isnan(context).any():
                continue
            ref_split = split_of(chunk_start)

            # STEP 3: Search — ALWAYS use raw_arrays (only match against REAL data)
            max_stop = min(n_arr, chunk_start) - chunk_len + 1
            pre_starts = np.arange(seasonal_window, max_stop, max(1, chunk_len // 4))
            pre_starts = [i for i in pre_starts if window_allowed(i, chunk_len, ref_split)]

            best_score, best_i = -np.inf, None
            for i in pre_starts:
                hc = raw_arrays[target_col][i - seasonal_window:i]
                hf = raw_arrays[target_col][i:i + chunk_len]
                if np.isnan(hc).any() or np.isnan(hf).any():
                    continue
                c = np.corrcoef(context, hc)[0, 1]
                if c > best_score:
                    best_score, best_i = c, i

            # Forward fallback — applied PER CHUNK, rarely needed
            if best_i is None:
                post_start = chunk_start + chunk_len + seasonal_window
                if post_start + chunk_len <= n_arr:
                    post_starts = np.arange(post_start, n_arr - chunk_len - seasonal_window + 1, max(1, chunk_len // 4))
                    post_starts = [i for i in post_starts if window_allowed(i, chunk_len, ref_split)]
                    for i in post_starts:
                        hc = raw_arrays[target_col][i - seasonal_window:i]
                        hf = raw_arrays[target_col][i:i + chunk_len]
                        if np.isnan(hc).any() or np.isnan(hf).any():
                            continue
                        c = np.corrcoef(context, hc)[0, 1]
                        if c > best_score:
                            best_score, best_i = c, i

            # STEP 4 & 5: Copy + scale
            if best_i is not None:
                for col in [c for c in col_names if c not in DETERMINISTIC_COLS]:
                    col_future = raw_arrays[col][best_i:best_i + chunk_len]    # always copy from REAL data
                    if len(col_future) == 0:
                        continue
                    cur_valid = arrays[col][context_start:chunk_start]         # chains from the previous chunk
                    mat_valid = raw_arrays[col][best_i - seasonal_window:best_i]
                    cur_valid = cur_valid[~np.isnan(cur_valid)]
                    mat_valid = mat_valid[~np.isnan(mat_valid)]

                    if col == "temp":
                        offset = (cur_valid.mean() - mat_valid.mean()) if len(cur_valid) and len(mat_valid) else 0.0
                        arrays[col][chunk_positions] = col_future + offset
                    else:
                        scale = (cur_valid.mean() / (mat_valid.mean() + 1e-6)) if len(cur_valid) and len(mat_valid) else 1.0
                        if col in ("rain_1h", "snow_1h"):
                            scale = 1.0
                        arrays[col][chunk_positions] = col_future * scale

    for col in col_names:
        d[col] = pd.Series(arrays[col], index=d.index)

    # Categorical columns: ffill, not linear interpolation
    cat_cols = [c for c in ("weather_type", "isSun") if c in d.columns]
    cat_backup = {c: d[c].copy() for c in cat_cols}
    d = d.interpolate(method="time").ffill().bfill()
    for c, orig in cat_backup.items():
        d[c] = orig.ffill().bfill()

    d[target_col] = d[target_col].clip(lower=0)

    # Hard physical bounds for columns prone to scaling drift
    for col, (lo, hi) in {"clouds_all": (0, 100), "humidity": (0, 100)}.items():
        if col in d.columns:
            d[col] = d[col].clip(lo, hi)

    # orig_max computed only from TRAIN to avoid leakage
    orig_max = (df_in.loc[df_in["Time"] <= train_cutoff_time, target_col].max()
                if train_cutoff_time is not None else df_in[target_col].max())
    d[target_col] = d[target_col].clip(lower=0, upper=orig_max)

    # Restore original dtypes + clip to valid ranges
    int_clip = {"isSun": (0, 1), "weather_type": (1, 5), "hour": (0, 23), "month": (1, 12),
                "clouds_all": (0, 100), "humidity": (0, 100)}
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

# ============================================================
# 2b. FILL GAPS ON ENTIRE DATASET BEFORE SPLITTING
# ============================================================

def run_preprocessing():
    """Load, gap-fill, split, clip, and engineer features. Returns (train_fe, val_fe, test_fe, train_pp, val_pp, test_pp)."""
    import pickle
    n_raw = len(df)
    prov_train_end = df.iloc[int(n_raw * 0.70) - 1]["Time"]
    prov_val_end = df.iloc[int(n_raw * 0.85) - 1]["Time"]

    print(f"\nFilling gaps on full dataset ({len(df):,} rows)...")
    df_filled, full_missing = reindex_fill(
        df, train_cutoff_time=prov_train_end, val_cutoff_time=prov_val_end
    )
    print(f"Filled dataset: {len(df_filled):,} rows, {full_missing.sum():,} gaps filled")

    def format_time_col(d):
        ts = d.index if isinstance(d.index, pd.DatetimeIndex) else pd.to_datetime(d["Time"])
        if isinstance(ts, pd.Series):
            return ts.dt.strftime("%m-%d-%Y %H:%M")
        return ts.strftime("%m-%d-%Y %H:%M")

    def format_cols(df):
        d = df.copy()
        float_prec = {"GHI": 1, "temp": 1, "wind_speed": 1, "rain_1h": 2, "snow_1h": 2, "SunlightTime/daylength": 2}
        for col, prec in float_prec.items():
            if col in d.columns:
                d[col] = d[col].round(prec)
        int_cols = ["Energy delta[Wh]", "pressure", "humidity", "clouds_all", "isSun",
                    "sunlightTime", "dayLength", "weather_type", "hour", "month"]
        for col in int_cols:
            if col in d.columns:
                d[col] = d[col].round().astype(int)
        return d

    def save_csv(df, path):
        out = df.reset_index()
        out["Time"] = format_time_col(out)
        out = format_cols(out)
        out.to_csv(path, index=False)
        print(f"Saved {path} ({len(out):,} rows)")

    save_csv(df_filled, "data/filled_renewable.csv")

    n = len(df_filled)
    train_end_idx = int(n * 0.70)
    val_end_idx = int(n * 0.85)
    train_end_time = df_filled.index[train_end_idx - 1]
    val_end_time = df_filled.index[val_end_idx - 1]
    print(f"Split indices (after fill): train_end={train_end_idx}, val_end={val_end_idx}")

    train_pp = df_filled.loc[:train_end_time].copy()
    val_pp = df_filled.loc[train_end_time + pd.Timedelta("15min"):val_end_time].copy()
    test_pp = df_filled.loc[val_end_time + pd.Timedelta("15min"):].copy()
    train_missing = full_missing.loc[:train_end_time]
    val_missing = full_missing.loc[train_end_time + pd.Timedelta("15min"):val_end_time]
    test_missing = full_missing.loc[val_end_time + pd.Timedelta("15min"):]
    print(f"Train after fill: {len(train_pp):,} rows ({train_missing.sum():,} filled)")
    print(f"Val after fill:   {len(val_pp):,} rows ({val_missing.sum():,} filled)")
    print(f"Test after fill:  {len(test_pp):,} rows ({test_missing.sum():,} filled)")

    for name, d in [("training", train_pp), ("validation", val_pp), ("testing", test_pp)]:
        save_csv(d, f"data/{name}.csv")

    for name, d, mask in [("training", train_pp, train_missing),
                           ("validation", val_pp, val_missing),
                           ("testing", test_pp, test_missing)]:
        filled_rows = d[mask].copy()
        save_csv(filled_rows, f"data/{name}_filled_gaps.csv")

    # Outlier clipping
    clip_config = {
        "GHI": (0.01, 0.99),
        "temp": (0.01, 0.99),
        "wind_speed": (0.01, 0.99),
        "rain_1h": (0.01, 0.99),
        "snow_1h": (0.01, 0.99),
        "clouds_all": (0.01, 0.99),
        "humidity": (0.01, 0.99),
        "pressure": (0.01, 0.99),
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

    # Feature engineering
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    def engineer(df_in, target_col=TARGET):
        d = df_in.copy()
        d["ED_lag1"] = d[target_col].shift(1)
        d["ED_lag4"] = d[target_col].shift(4)
        d["ED_lag96"] = d[target_col].shift(96)
        d["GHI_lag1"] = d["GHI"].shift(1)
        d["GHI_roll4"] = d["GHI"].shift(1).rolling(window=4, min_periods=1).mean()
        d["ED_roll4"] = d[target_col].shift(1).rolling(window=4, min_periods=1).mean()
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

    print("\nPREPROCESSING COMPLETE\n")
    return train_fe, val_fe, test_fe, train_pp, val_pp, test_pp

if __name__ == "__main__":
    train_fe, val_fe, test_fe, train_pp, val_pp, test_pp = run_preprocessing()
    import pickle
    for name, obj in [("train_fe", train_fe), ("val_fe", val_fe), ("test_fe", test_fe),
                       ("train_pp", train_pp), ("val_pp", val_pp), ("test_pp", test_pp)]:
        with open(f"data/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    print("Saved engineered data to pickle files.")
