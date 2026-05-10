"""ROGII v4: v3 features + LGB + CatBoost blend (45/55) + postproc (alpha=0.9, tau=250, SavGol win=101).

Expected OOF ~14.76 → projected LB ~13.9 (vs v3 LB 14.521).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.signal import savgol_filter

# locate comp data
ROOT = None
for p in Path("/kaggle/input").rglob("sample_submission.csv"):
    ROOT = p.parent; break
if ROOT is None:
    sys.exit("error: comp data not mounted")
print(f"comp root: {ROOT}", flush=True)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
NCC_WINDOWS = (11, 25, 51)
SEARCH_RADIUS_FT = 30


def detect_ps(h):
    nan_mask = h["TVT_input"].isna()
    return int(nan_mask.idxmax()) if nan_mask.any() else len(h)


def fillna_gr(arr):
    return pd.Series(arr).interpolate(limit_direction="both").to_numpy()


def _interp(t_tvt, t_gr, x):
    return np.interp(x, t_tvt, t_gr)


def _sliding_windows(arr, win):
    if len(arr) < win:
        return np.empty((0, win), dtype=arr.dtype)
    return np.lib.stride_tricks.sliding_window_view(arr, win)


def build_v1(h, t, well, include_target):
    ps = detect_ps(h)
    if ps == 0 or ps >= len(h):
        return pd.DataFrame()
    h_gr_full = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy()
    t_gr = fillna_gr(t["GR"].to_numpy())
    if not np.all(np.diff(t_tvt) >= 0):
        order = np.argsort(t_tvt); t_tvt = t_tvt[order]; t_gr = t_gr[order]
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
    s_gr = pd.Series(h_gr_full)
    rmean_21 = s_gr.rolling(21, min_periods=1, center=True).mean().to_numpy()
    rstd_21 = s_gr.rolling(21, min_periods=1, center=True).std().to_numpy()
    rmean_51 = s_gr.rolling(51, min_periods=1, center=True).mean().to_numpy()
    rstd_51 = s_gr.rolling(51, min_periods=1, center=True).std().to_numpy()
    rmean_101 = s_gr.rolling(101, min_periods=1, center=True).mean().to_numpy()
    out = {
        "well": well, "row_idx": np.arange(ps, len(h)),
        "MD": post["MD"].to_numpy(), "X": post["X"].to_numpy(),
        "Y": post["Y"].to_numpy(), "Z": post["Z"].to_numpy(),
        "hw_gr": h_gr_full[ps:],
        "rmean_21": rmean_21[ps:], "rstd_21": rstd_21[ps:],
        "rmean_51": rmean_51[ps:], "rstd_51": rstd_51[ps:],
        "rmean_101": rmean_101[ps:],
        "md_off": post["MD"].to_numpy() - md_at_ps,
        "xy_dist": np.sqrt((post["X"].to_numpy() - x_at_ps) ** 2
                           + (post["Y"].to_numpy() - y_at_ps) ** 2),
        "z_off": post["Z"].to_numpy() - z_at_ps,
        "last_known_tvt": last_known_tvt, "last_dtvt": last_dtvt,
        "cal_len": ps, "cal_gr_rmse": cal_gr_rmse,
    }
    offsets = list(range(-15, 16, 3))
    lk_arr = np.full(n_post, last_known_tvt)
    for k in offsets:
        out[f"tw_gr_off_{k:+d}"] = _interp(t_tvt, t_gr, lk_arr + k)
    out["gr_diff"] = out["hw_gr"] - out["tw_gr_off_+0"]
    if include_target:
        out["target"] = post["TVT"].to_numpy() - last_known_tvt
    return pd.DataFrame(out)


def add_ncc(df, h, t, ps):
    h_gr_full = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy(); t_gr = fillna_gr(t["GR"].to_numpy())
    if not np.all(np.diff(t_tvt) >= 0):
        order = np.argsort(t_tvt); t_tvt = t_tvt[order]; t_gr = t_gr[order]
    last_known_tvt = float(h["TVT_input"].iloc[ps - 1])
    n_post = len(df)
    lo_tvt = last_known_tvt - SEARCH_RADIUS_FT; hi_tvt = last_known_tvt + SEARCH_RADIUS_FT
    cand_lo = int(np.searchsorted(t_tvt, lo_tvt, side="left"))
    cand_hi = int(np.searchsorted(t_tvt, hi_tvt, side="right"))
    if cand_hi - cand_lo < max(NCC_WINDOWS):
        cand_lo = max(0, cand_lo - max(NCC_WINDOWS))
        cand_hi = min(len(t_tvt), cand_hi + max(NCC_WINDOWS))
    t_gr_cand = t_gr[cand_lo:cand_hi]; t_tvt_cand = t_tvt[cand_lo:cand_hi]
    lk_pos_in_region = int(np.clip(np.searchsorted(t_tvt_cand, last_known_tvt), 0, len(t_tvt_cand) - 1))
    for W in NCC_WINDOWS:
        half = W // 2
        tw_wins = _sliding_windows(t_gr_cand, W)
        tw_centre_tvt = t_tvt_cand[half: half + tw_wins.shape[0]]
        if tw_wins.shape[0] == 0:
            df[f"ncc_pred_tvt_w{W}"] = last_known_tvt
            df[f"ncc_pred_dlt_w{W}"] = 0.0
            df[f"ncc_score_w{W}"] = np.nan
            df[f"ncc_score_w{W}_at_lk"] = np.nan
            df[f"ncc_score_ratio_w{W}"] = 1.0
            continue
        h_indices = df["row_idx"].to_numpy()
        ncc_pred_tvt = np.full(n_post, last_known_tvt, dtype=np.float64)
        ncc_score = np.full(n_post, np.nan, dtype=np.float64)
        ncc_score_at_lk = np.full(n_post, np.nan, dtype=np.float64)
        lk_win_idx = lk_pos_in_region - half
        lk_tw_win = tw_wins[lk_win_idx] if 0 <= lk_win_idx < tw_wins.shape[0] else tw_wins[tw_wins.shape[0] // 2]
        for ii, hi_global in enumerate(h_indices):
            lo = hi_global - half; hi = hi_global + half + 1
            if lo < 0 or hi > len(h_gr_full): continue
            h_win = h_gr_full[lo:hi]
            if not np.isfinite(h_win).all(): continue
            d = np.mean((tw_wins - h_win) ** 2, axis=1)
            best = int(np.argmin(d))
            ncc_pred_tvt[ii] = tw_centre_tvt[best]
            ncc_score[ii] = d[best]
            ncc_score_at_lk[ii] = float(np.mean((lk_tw_win - h_win) ** 2))
        df[f"ncc_pred_tvt_w{W}"] = ncc_pred_tvt
        df[f"ncc_pred_dlt_w{W}"] = ncc_pred_tvt - last_known_tvt
        df[f"ncc_score_w{W}"] = ncc_score
        df[f"ncc_score_w{W}_at_lk"] = ncc_score_at_lk
        df[f"ncc_score_ratio_w{W}"] = ncc_score / np.where(ncc_score_at_lk > 0, ncc_score_at_lk, 1.0)


def build_centroids(train_dir):
    rows = []
    for hp in sorted(train_dir.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        h = pd.read_csv(hp)
        if not all(f in h.columns for f in FORMATIONS): continue
        rec = {"well": well, "X": float(h["X"].median()), "Y": float(h["Y"].median())}
        for f in FORMATIONS:
            rec[f"{f.lower()}_z"] = float(h[f].median())
        rows.append(rec)
    return pd.DataFrame(rows)


def estimate_formations(xy_query, centroids, exclude_well=None, k=10, eps=1e-6):
    c = centroids[centroids["well"] != exclude_well].reset_index(drop=True) if exclude_well else centroids
    cxy = c[["X", "Y"]].values
    out = {f.lower(): np.full(len(xy_query), np.nan) for f in FORMATIONS}
    if len(c) == 0: return out
    diff = xy_query[:, None, :] - cxy[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    K = min(k, d2.shape[1])
    idx = np.argpartition(d2, K - 1, axis=1)[:, :K]
    rows = np.arange(len(xy_query))[:, None]
    nn_d2 = d2[rows, idx]
    w = 1.0 / (nn_d2 + eps)
    w_sum = w.sum(axis=1, keepdims=True)
    for f in FORMATIONS:
        col = f"{f.lower()}_z"
        vals = c[col].values[idx]
        out[f.lower()] = (vals * w).sum(axis=1) / w_sum.flatten()
    return out


def add_formation_features(df, centroids, well, is_train):
    xy = df[["X", "Y"]].values
    exclude = well if is_train else None
    est = estimate_formations(xy, centroids, exclude_well=exclude, k=10)
    z_arr = df["Z"].to_numpy(); lk_arr = df["last_known_tvt"].to_numpy()
    for f in FORMATIONS:
        key = f.lower()
        df[f"{key}_est"] = est[key]
        df[f"z_to_{key}"] = z_arr - est[key]
        df[f"lk_to_{key}"] = lk_arr - est[key]


def build_dataset(split_dir, centroids, is_train, include_target):
    parts = []
    for hp in sorted(split_dir.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        tp = split_dir / f"{well}__typewell.csv"
        if not tp.exists(): continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_v1(h, t, well, include_target)
        if len(df) == 0: continue
        ps = detect_ps(h)
        add_ncc(df, h, t, ps)
        add_formation_features(df, centroids, well, is_train)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


print("=== build train centroids ===", flush=True)
t0 = time.time()
centroids = build_centroids(ROOT / "train")
print(f"centroids: {centroids.shape} in {time.time()-t0:.0f}s", flush=True)

print("\n=== build TRAIN features ===", flush=True)
t0 = time.time()
train_df = build_dataset(ROOT / "train", centroids, is_train=True, include_target=True)
print(f"train: {len(train_df)} rows, {train_df['well'].nunique()} wells in {time.time()-t0:.0f}s", flush=True)

print("=== build TEST features ===", flush=True)
test_df = build_dataset(ROOT / "test", centroids, is_train=False, include_target=False)
print(f"test: {len(test_df)} rows, {test_df['well'].nunique()} wells", flush=True)

feat_cols = [c for c in train_df.columns if c not in ("well", "row_idx", "target")]
print(f"features: {len(feat_cols)}", flush=True)

X_tr = train_df[feat_cols].values
y_tr = train_df["target"].values
X_te = test_df[feat_cols].values

print("\n=== train LGB ===", flush=True)
t0 = time.time()
m_lgb = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=42,
)
m_lgb.fit(X_tr, y_tr)
print(f"  trained in {time.time()-t0:.0f}s", flush=True)

print("=== train CatBoost ===", flush=True)
t0 = time.time()
m_cat = CatBoostRegressor(
    iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
    loss_function="RMSE", random_seed=42, thread_count=-1,
    allow_writing_files=False, verbose=0,
)
m_cat.fit(X_tr, y_tr)
print(f"  trained in {time.time()-t0:.0f}s", flush=True)

print("=== predict + blend (45/55) ===", flush=True)
pred_lgb = m_lgb.predict(X_te)
pred_cat = m_cat.predict(X_te)
W_LGB = 0.45
delta = W_LGB * pred_lgb + (1 - W_LGB) * pred_cat

# postproc: alpha shrink + tau fade-in
ALPHA = 0.9; TAU = 250
md_off = test_df["md_off"].values
factor = 1.0 - np.exp(-md_off / TAU)
delta = delta * ALPHA * factor

test_df["tvt"] = test_df["last_known_tvt"] + delta

# postproc: per-well Savitzky-Golay (win=101, ord=3)
SG_WIN = 101; SG_ORD = 3
for w in test_df["well"].unique():
    mask = test_df["well"].values == w
    n = mask.sum()
    if n >= SG_WIN:
        test_df.loc[mask, "tvt"] = savgol_filter(test_df.loc[mask, "tvt"].values,
                                                  window_length=SG_WIN, polyorder=SG_ORD)

# Build submission
sub = pd.read_csv(ROOT / "sample_submission.csv")
test_df["id"] = test_df["well"] + "_" + test_df["row_idx"].astype(str)
lookup = test_df.set_index("id")["tvt"]
sub["tvt"] = sub["id"].map(lookup)
n_missing = sub["tvt"].isna().sum()
if n_missing > 0:
    print(f"WARN: {n_missing} ids missing", flush=True)
    sub["tvt"] = sub["tvt"].fillna(test_df["last_known_tvt"].median())

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: shape={sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
