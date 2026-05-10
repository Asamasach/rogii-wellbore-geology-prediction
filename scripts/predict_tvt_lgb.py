"""LightGBM residual predictor: target = TVT - last_known_tvt.

Builds a flat per-row dataset (well, MD, post-PS only) with simple but
high-signal features:
  - hw_gr: gamma ray at this row
  - tw_gr_lk: typewell GR at last_known_tvt (the 'expected GR if flat')
  - gr_diff: hw_gr - tw_gr_lk (the residual GR)
  - tw_gr_off_<k>: typewell GR at last_known_tvt + k for k in [-15..15]
  - rolling stats of hw_gr in windows {21, 51, 101}
  - md_off: MD - MD_at_PS (how far into the prediction zone)
  - xy_dist: euclidean distance in (X,Y) from row at PS-1
  - z_off:   Z - Z_at_PS-1
  - last_known_tvt
  - last_dtvt: dTVT/dMD slope at end of calibration (linear fit on last 50)
  - cal_len: length of calibration zone
  - cal_gr_rmse: residual RMSE in calibration between hw_gr and tw_gr_at(TVT_input)

Target: TVT - last_known_tvt (i.e. how far TVT has moved from its last known
value). Final prediction = last_known_tvt + lgb_predicted_delta.

GroupKFold(5) by well for OOF, then refit on all train for test.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import time
import numpy as np
import pandas as pd

from predict_tvt import detect_ps, fillna_gr


def _interp(t_tvt, t_gr, x):
    return np.interp(x, t_tvt, t_gr)


def build_features_for_well(h: pd.DataFrame, t: pd.DataFrame, well: str,
                            include_target: bool = True) -> pd.DataFrame:
    """Build a per-row dataframe (for rows >= PS). Returns features + target columns.

    If include_target=False (test), the 'target' column is omitted/NaN.
    """
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

    # Calibration-zone diagnostics
    pre_tvt = h["TVT_input"].iloc[:ps].to_numpy()
    pre_gr = h_gr_full[:ps]
    expected_gr_pre = _interp(t_tvt, t_gr, pre_tvt)
    resid_pre = pre_gr - expected_gr_pre
    cal_gr_rmse = float(np.sqrt(np.nanmean(resid_pre ** 2))) if len(resid_pre) > 0 else np.nan

    # Last-50 slope (dTVT/dMD)
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

    # Position at PS-1
    md_at_ps = float(h["MD"].iloc[ps - 1])
    x_at_ps = float(h["X"].iloc[ps - 1]); y_at_ps = float(h["Y"].iloc[ps - 1])
    z_at_ps = float(h["Z"].iloc[ps - 1])

    # Slice post-PS rows
    post = h.iloc[ps:].reset_index(drop=True)
    n_post = len(post)

    # Rolling stats need to include some pre-PS context — pad with calibration
    # zone for the full series to be valid at PS.
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
    # Typewell GR at last_known_tvt + offset (k in -15..15 by 3)
    offsets = list(range(-15, 16, 3))
    lk_arr = np.full(n_post, last_known_tvt)
    for k in offsets:
        out[f"tw_gr_off_{k:+d}"] = _interp(t_tvt, t_gr, lk_arr + k)
    out["gr_diff"] = out["hw_gr"] - out["tw_gr_off_+0"]

    if include_target:
        out["target"] = post["TVT"].to_numpy() - last_known_tvt
        out["true_tvt"] = post["TVT"].to_numpy()

    return pd.DataFrame(out)


def build_dataset(train_dir: Path, max_wells: int | None = None):
    """Build the flattened per-row train dataset across all wells."""
    files = sorted(train_dir.glob("*horizontal_well.csv"))
    if max_wells:
        files = files[:max_wells]
    parts = []
    for i, hp in enumerate(files):
        well = hp.name.split("__")[0]
        tp = train_dir / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well(h, t, well, include_target=True)
        if len(df):
            parts.append(df)
        if (i + 1) % 100 == 0:
            print(f"  built features {i+1}/{len(files)}", flush=True)
    return pd.concat(parts, ignore_index=True)


# --------------------- driver: small local test --------------------------
if __name__ == "__main__":
    SAMPLES = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_train")
    print("=== building features for 4 local sample wells ===")
    parts = []
    for hp in sorted(SAMPLES.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        tp = SAMPLES / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well(h, t, well, include_target=True)
        parts.append(df)
        print(f"  {well}: {df.shape}")
    full = pd.concat(parts, ignore_index=True)
    print(f"\ntotal rows: {len(full)}")
    print(f"columns ({len(full.columns)}): {list(full.columns)[:25]}")
    print(f"\ntarget stats: mean={full['target'].mean():.3f}  std={full['target'].std():.3f}  "
          f"min={full['target'].min():.2f}  max={full['target'].max():.2f}")
    print(f"\nhead:\n{full.head(3).to_string(index=False)[:600]}")

    # Quick LGB sanity (single fold, no GroupKFold yet)
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    feat_cols = [c for c in full.columns if c not in ("well", "row_idx", "target", "true_tvt")]
    X = full[feat_cols].values
    y = full["target"].values
    print(f"\n{len(feat_cols)} features, {len(full)} rows")

    # 50/50 split (no group-aware here, just sanity)
    n = len(full)
    half = n // 2
    Xtr, ytr = X[:half], y[:half]
    Xva, yva = X[half:], y[half:]
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50)])
    pred_delta = m.predict(Xva)
    pred_tvt = full["last_known_tvt"].iloc[half:].values + pred_delta
    true_tvt = full["true_tvt"].iloc[half:].values
    rmse = float(np.sqrt(np.mean((true_tvt - pred_tvt) ** 2)))
    base_rmse = float(np.sqrt(np.mean((true_tvt - full["last_known_tvt"].iloc[half:].values) ** 2)))
    print(f"\nbaseline (const) RMSE on val half: {base_rmse:.3f}")
    print(f"LGB RMSE on val half:              {rmse:.3f}")
