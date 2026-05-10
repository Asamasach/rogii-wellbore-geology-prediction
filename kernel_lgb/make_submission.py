"""ROGII LGB v1 submission.

End-to-end: build features for all 773 train wells + 3 test wells, train
LightGBM on residual = TVT - last_known_tvt, predict on test, write
/kaggle/working/submission.csv.

Self-contained — inlines feature engineering and the const + lgb predictor.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# auto-detect comp root
ROOT = None
for p in Path("/kaggle/input").rglob("sample_submission.csv"):
    ROOT = p.parent
    break
if ROOT is None:
    sys.exit("error: comp data not mounted")
print(f"comp root: {ROOT}", flush=True)


def detect_ps(h: pd.DataFrame) -> int:
    nan_mask = h["TVT_input"].isna()
    return int(nan_mask.idxmax()) if nan_mask.any() else len(h)


def fillna_gr(arr):
    return pd.Series(arr).interpolate(limit_direction="both").to_numpy()


def _interp(t_tvt, t_gr, x):
    return np.interp(x, t_tvt, t_gr)


def build_features_for_well(h: pd.DataFrame, t: pd.DataFrame, well: str,
                            include_target: bool) -> pd.DataFrame:
    ps = detect_ps(h)
    if ps == 0 or ps >= len(h):
        return pd.DataFrame()

    h_gr_full = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy()
    t_gr = fillna_gr(t["GR"].to_numpy())
    if not np.all(np.diff(t_tvt) >= 0):
        order = np.argsort(t_tvt)
        t_tvt = t_tvt[order]; t_gr = t_gr[order]

    last_known_tvt = float(h["TVT_input"].iloc[ps - 1])
    pre_tvt = h["TVT_input"].iloc[:ps].to_numpy()
    pre_gr = h_gr_full[:ps]
    expected_gr_pre = _interp(t_tvt, t_gr, pre_tvt)
    resid_pre = pre_gr - expected_gr_pre
    cal_gr_rmse = float(np.sqrt(np.nanmean(resid_pre ** 2))) if len(resid_pre) > 0 else np.nan

    if ps >= 5:
        n = min(50, ps - 1)
        md_w = h["MD"].iloc[ps - 1 - n: ps].to_numpy()
        tvt_w = h["TVT_input"].iloc[ps - 1 - n: ps].to_numpy()
        if len(md_w) >= 2 and np.ptp(md_w) > 0:
            last_dtvt = float(np.polyfit(md_w, tvt_w, 1)[0])
        else:
            last_dtvt = 0.0
    else:
        last_dtvt = 0.0

    md_at_ps = float(h["MD"].iloc[ps - 1])
    x_at_ps = float(h["X"].iloc[ps - 1]); y_at_ps = float(h["Y"].iloc[ps - 1])
    z_at_ps = float(h["Z"].iloc[ps - 1])

    post = h.iloc[ps:].reset_index(drop=True)
    n_post = len(post)

    full_gr = h_gr_full
    s_gr = pd.Series(full_gr)
    rmean_21 = s_gr.rolling(21, min_periods=1, center=True).mean().to_numpy()
    rstd_21 = s_gr.rolling(21, min_periods=1, center=True).std().to_numpy()
    rmean_51 = s_gr.rolling(51, min_periods=1, center=True).mean().to_numpy()
    rstd_51 = s_gr.rolling(51, min_periods=1, center=True).std().to_numpy()
    rmean_101 = s_gr.rolling(101, min_periods=1, center=True).mean().to_numpy()

    out = {
        "well": well,
        "row_idx": np.arange(ps, len(h)),
        "MD": post["MD"].to_numpy(),
        "X": post["X"].to_numpy(),
        "Y": post["Y"].to_numpy(),
        "Z": post["Z"].to_numpy(),
        "hw_gr": h_gr_full[ps:],
        "rmean_21": rmean_21[ps:],
        "rstd_21": rstd_21[ps:],
        "rmean_51": rmean_51[ps:],
        "rstd_51": rstd_51[ps:],
        "rmean_101": rmean_101[ps:],
        "md_off": post["MD"].to_numpy() - md_at_ps,
        "xy_dist": np.sqrt((post["X"].to_numpy() - x_at_ps) ** 2
                           + (post["Y"].to_numpy() - y_at_ps) ** 2),
        "z_off": post["Z"].to_numpy() - z_at_ps,
        "last_known_tvt": last_known_tvt,
        "last_dtvt": last_dtvt,
        "cal_len": ps,
        "cal_gr_rmse": cal_gr_rmse,
    }
    offsets = list(range(-15, 16, 3))
    lk_arr = np.full(n_post, last_known_tvt)
    for k in offsets:
        out[f"tw_gr_off_{k:+d}"] = _interp(t_tvt, t_gr, lk_arr + k)
    out["gr_diff"] = out["hw_gr"] - out["tw_gr_off_+0"]

    if include_target:
        out["target"] = post["TVT"].to_numpy() - last_known_tvt

    return pd.DataFrame(out)


def build_dataset(split_dir: Path, include_target: bool):
    files = sorted(split_dir.glob("*horizontal_well.csv"))
    parts = []
    for hp in files:
        well = hp.name.split("__")[0]
        tp = split_dir / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well(h, t, well, include_target=include_target)
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


print("=== build TRAIN features ===", flush=True)
t0 = time.time()
train_df = build_dataset(ROOT / "train", include_target=True)
print(f"train: {len(train_df)} rows, {train_df['well'].nunique()} wells in {time.time()-t0:.0f}s",
      flush=True)

print("=== build TEST features ===", flush=True)
test_df = build_dataset(ROOT / "test", include_target=False)
print(f"test: {len(test_df)} rows, {test_df['well'].nunique()} wells", flush=True)

feat_cols = [c for c in train_df.columns
             if c not in ("well", "row_idx", "target")]
print(f"features: {len(feat_cols)}", flush=True)

print("=== train LGB on full train ===", flush=True)
t0 = time.time()
m = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=42,
)
m.fit(train_df[feat_cols].values, train_df["target"].values)
print(f"trained in {time.time()-t0:.0f}s", flush=True)

print("=== predict on test ===", flush=True)
test_df["pred_delta"] = m.predict(test_df[feat_cols].values)
test_df["tvt"] = test_df["last_known_tvt"] + test_df["pred_delta"]

# Build submission in correct order from sample_submission
sub = pd.read_csv(ROOT / "sample_submission.csv")
# Map (well, row_idx) -> tvt
test_df["id"] = test_df["well"] + "_" + test_df["row_idx"].astype(str)
lookup = test_df.set_index("id")["tvt"]
sub["tvt"] = sub["id"].map(lookup)
n_missing = sub["tvt"].isna().sum()
if n_missing > 0:
    print(f"WARN: {n_missing} ids missing prediction; filling with median last_known", flush=True)
    sub["tvt"] = sub["tvt"].fillna(test_df["last_known_tvt"].median())

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: shape={sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
