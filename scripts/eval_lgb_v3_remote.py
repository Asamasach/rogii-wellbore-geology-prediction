"""GroupKFold(5) eval of LGB v3 (formation plane-fit features)."""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_tvt_lgb_v3 import build_centroids, build_dataset_v3

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/data/raw")
TRAIN_DIR = ROOT / "train"
LOG_DIR = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=== build train centroids (773 wells, 6 formations each) ===", flush=True)
t0 = time.time()
centroids = build_centroids(TRAIN_DIR)
print(f"centroids: {centroids.shape} in {time.time()-t0:.1f}s", flush=True)

print(f"\n=== build LGB v3 feature set ===", flush=True)
t0 = time.time()
full = build_dataset_v3(TRAIN_DIR, centroids, is_train=True)
print(f"\nbuilt {len(full)} rows from {full['well'].nunique()} wells in {time.time()-t0:.0f}s", flush=True)

feat_cols = [c for c in full.columns if c not in ("well", "row_idx", "target", "true_tvt")]
X = full[feat_cols].values
y = full["target"].values
groups = full["well"].values
last_known = full["last_known_tvt"].values
true_tvt = full["true_tvt"].values
print(f"features: {len(feat_cols)}", flush=True)

const_rmse = float(np.sqrt(np.mean((true_tvt - last_known) ** 2)))
print(f"const pooled RMSE: {const_rmse:.4f}", flush=True)

print(f"\n=== LGB v3 GroupKFold(5) by well ===", flush=True)
gkf = GroupKFold(n_splits=5)
oof_delta = np.zeros(len(full), dtype=np.float64)
fold_rmses = []
for fold, (tr_i, va_i) in enumerate(gkf.split(X, y, groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=42,
    )
    m.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_delta[va_i] = m.predict(X[va_i])
    fold_pred_tvt = last_known[va_i] + oof_delta[va_i]
    fold_rmse = float(np.sqrt(np.mean((true_tvt[va_i] - fold_pred_tvt) ** 2)))
    fold_const = float(np.sqrt(np.mean((true_tvt[va_i] - last_known[va_i]) ** 2)))
    fold_rmses.append(fold_rmse)
    print(f"  fold {fold}: const={fold_const:.4f}  lgb={fold_rmse:.4f}  best_iter={m.best_iteration_}  t={time.time()-t0:.0f}s", flush=True)

oof_pred_tvt = last_known + oof_delta
oof_rmse = float(np.sqrt(np.mean((true_tvt - oof_pred_tvt) ** 2)))
print(f"\nLGB v3 OOF pooled RMSE: {oof_rmse:.4f}", flush=True)
print(f"  vs v2 (15.36): {15.36 - oof_rmse:+.4f}", flush=True)
print(f"  vs v1 (15.31): {15.31 - oof_rmse:+.4f}", flush=True)
print(f"  vs const (15.91): {const_rmse - oof_rmse:+.4f}", flush=True)

np.save(LOG_DIR.parent / "models" / "oof" / "oof_delta_lgb_v3.npy", oof_delta)

print(f"\n=== feature importance (top 25) ===", flush=True)
fi = pd.DataFrame({"feat": feat_cols, "imp": m.feature_importances_}).sort_values("imp", ascending=False)
print(fi.head(25).to_string(index=False), flush=True)
