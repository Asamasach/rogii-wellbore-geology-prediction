"""Grid search post-processing on v3 OOF: alpha shrink + tau fade-in + per-well SavGol smoothing.

Operates on the saved OOF deltas from eval_lgb_v3_remote.py.

  delta_alpha     = oof_delta * alpha
  delta_alpha_tau = delta_alpha * (1 - exp(-md_since_ps / tau))
  pred_tvt        = last_known + delta_alpha_tau
  pred_tvt_smooth = savgol(pred_tvt, win=17, order=3) per well

Reports pooled RMSE for each (alpha, tau, smoothing) combination.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_tvt_lgb_v3 import build_centroids, build_dataset_v3

from scipy.signal import savgol_filter

ROOT = Path("/home/fteam11/projects/rogii-wellbore-geology-prediction/data/raw")
TRAIN_DIR = ROOT / "train"

print("=== load v3 OOF deltas ===", flush=True)
oof = np.load("/home/fteam11/projects/rogii-wellbore-geology-prediction/models/oof/oof_delta_lgb_v3.npy")
print(f"oof shape: {oof.shape}", flush=True)

print("=== rebuild v3 dataset (need feature columns for md_off, last_known_tvt, true_tvt) ===", flush=True)
centroids = build_centroids(TRAIN_DIR)
full = build_dataset_v3(TRAIN_DIR, centroids, is_train=True)
assert len(full) == len(oof), f"mismatch {len(full)} vs {len(oof)}"

last_known = full["last_known_tvt"].values
true_tvt = full["true_tvt"].values
md_off = full["md_off"].values
wells = full["well"].values

base_pred = last_known + oof
base_rmse = float(np.sqrt(np.mean((true_tvt - base_pred) ** 2)))
print(f"baseline (no postproc): {base_rmse:.4f}", flush=True)

print("\n=== alpha (shrink) sweep, no tau, no smoothing ===", flush=True)
for alpha in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
    p = last_known + oof * alpha
    rmse = float(np.sqrt(np.mean((true_tvt - p) ** 2)))
    print(f"  alpha={alpha:.2f}: {rmse:.4f}  delta={base_rmse - rmse:+.4f}", flush=True)

# tau fade-in: factor = 1 - exp(-md_off / tau)
print("\n=== alpha + tau (fade-in) sweep ===", flush=True)
best = (None, None, base_rmse)
for alpha in [1.0, 0.9, 0.85, 0.8]:
    for tau in [None, 30, 60, 120, 250, 500, 1000]:
        if tau is None:
            factor = 1.0
        else:
            factor = 1.0 - np.exp(-md_off / tau)
        p = last_known + oof * alpha * factor
        rmse = float(np.sqrt(np.mean((true_tvt - p) ** 2)))
        if rmse < best[2]:
            best = (alpha, tau, rmse)
        print(f"  alpha={alpha:.2f} tau={tau}: {rmse:.4f}  delta={base_rmse - rmse:+.4f}", flush=True)

print(f"\nBEST alpha+tau: alpha={best[0]} tau={best[1]} rmse={best[2]:.4f} (delta {base_rmse - best[2]:+.4f})", flush=True)

# Apply best alpha+tau then per-well Savitzky-Golay smoothing
alpha, tau, _ = best
factor = 1.0 if tau is None else 1.0 - np.exp(-md_off / tau)
post_pred = last_known + oof * alpha * factor

print("\n=== add per-well SavGol smoothing on top of best alpha+tau ===", flush=True)
for win, order in [(17, 3), (25, 3), (51, 3), (101, 3), (51, 4), (51, 5)]:
    smooth = post_pred.copy()
    for w in pd.unique(wells):
        mask = wells == w
        n = mask.sum()
        if n < win:
            continue
        smooth[mask] = savgol_filter(post_pred[mask], window_length=win, polyorder=order)
    rmse = float(np.sqrt(np.mean((true_tvt - smooth) ** 2)))
    print(f"  savgol win={win} order={order}: {rmse:.4f}  delta_vs_alpha_tau={best[2] - rmse:+.4f}",
          flush=True)
