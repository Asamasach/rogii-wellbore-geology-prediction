"""Train CatBoost on v3 features (GroupKFold 5), then Ridge-stack with LGB v3 OOF."""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_tvt_lgb_v3 import build_centroids, build_dataset_v3

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

ROOT = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/data/raw")
TRAIN_DIR = ROOT / "train"
LOG_DIR = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/logs")
OOF_DIR = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/models/oof")

print("=== load v3 dataset (rebuild) ===", flush=True)
t0 = time.time()
centroids = build_centroids(TRAIN_DIR)
full = build_dataset_v3(TRAIN_DIR, centroids, is_train=True)
print(f"built {len(full)} rows in {time.time()-t0:.0f}s", flush=True)

feat_cols = [c for c in full.columns if c not in ("well", "row_idx", "target", "true_tvt")]
X = full[feat_cols].values
y = full["target"].values
groups = full["well"].values
last_known = full["last_known_tvt"].values
true_tvt = full["true_tvt"].values
print(f"features: {len(feat_cols)}", flush=True)

print("\n=== CatBoost GroupKFold(5) ===", flush=True)
gkf = GroupKFold(n_splits=5)
oof_cat = np.zeros(len(full))
for fold, (tr_i, va_i) in enumerate(gkf.split(X, y, groups)):
    t0 = time.time()
    m = CatBoostRegressor(
        iterations=2000, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=42, od_type="Iter", od_wait=100,
        thread_count=-1, allow_writing_files=False, verbose=0,
    )
    m.fit(X[tr_i], y[tr_i], eval_set=(X[va_i], y[va_i]), use_best_model=True)
    oof_cat[va_i] = m.predict(X[va_i])
    fold_pred = last_known[va_i] + oof_cat[va_i]
    fold_rmse = float(np.sqrt(np.mean((true_tvt[va_i] - fold_pred) ** 2)))
    print(f"  fold {fold}: cat={fold_rmse:.4f}  best_iter={m.tree_count_}  t={time.time()-t0:.0f}s",
          flush=True)
np.save(OOF_DIR / "oof_delta_cat_v3.npy", oof_cat)
cat_pred = last_known + oof_cat
cat_rmse = float(np.sqrt(np.mean((true_tvt - cat_pred) ** 2)))
print(f"\nCatBoost v3 OOF pooled RMSE: {cat_rmse:.4f}  (LGB v3 was 14.97)", flush=True)

print("\n=== load LGB v3 OOF + Ridge stack ===", flush=True)
oof_lgb = np.load(OOF_DIR / "oof_delta_lgb_v3.npy")
lgb_pred = last_known + oof_lgb
lgb_rmse = float(np.sqrt(np.mean((true_tvt - lgb_pred) ** 2)))
print(f"  LGB OOF RMSE: {lgb_rmse:.4f}", flush=True)
print(f"  CAT OOF RMSE: {cat_rmse:.4f}", flush=True)

# Simple average
avg_pred = last_known + (oof_lgb + oof_cat) / 2
avg_rmse = float(np.sqrt(np.mean((true_tvt - avg_pred) ** 2)))
print(f"  AVG (LGB+CAT)/2: {avg_rmse:.4f}", flush=True)

# Ridge stack — predict target from (lgb_delta, cat_delta), GroupKFold leak-free
stack_X = np.column_stack([oof_lgb, oof_cat])
stack_oof = np.zeros(len(full))
for tr_i, va_i in gkf.split(stack_X, y, groups):
    r = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    r.fit(stack_X[tr_i], y[tr_i])
    stack_oof[va_i] = r.predict(stack_X[va_i])
stack_pred = last_known + stack_oof
stack_rmse = float(np.sqrt(np.mean((true_tvt - stack_pred) ** 2)))
print(f"  Ridge stack OOF: {stack_rmse:.4f}", flush=True)

# Also fit a SINGLE ridge on full to get the stack weights for production
r_full = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(stack_X, y)
print(f"  Ridge stack final weights: lgb={r_full.coef_[0]:.4f}  cat={r_full.coef_[1]:.4f}", flush=True)

# Best weighted blend search (exhaustive 0..1 step 0.05)
print("\n=== exhaustive blend search ===", flush=True)
best = (None, np.inf)
for w_lgb in np.arange(0.0, 1.01, 0.05):
    blend = w_lgb * oof_lgb + (1 - w_lgb) * oof_cat
    pred = last_known + blend
    r = float(np.sqrt(np.mean((true_tvt - pred) ** 2)))
    if r < best[1]:
        best = (w_lgb, r)
print(f"BEST blend: w_lgb={best[0]:.2f} w_cat={1-best[0]:.2f} OOF RMSE={best[1]:.4f}", flush=True)

# Save best blend OOF for downstream use
best_w = best[0]
np.save(OOF_DIR / "oof_delta_blend_v3.npy", best_w * oof_lgb + (1 - best_w) * oof_cat)
print(f"saved oof_delta_blend_v3.npy", flush=True)
