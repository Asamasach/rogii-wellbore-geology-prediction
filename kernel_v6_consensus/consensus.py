"""ROGII v6 — multi-fork consensus with uncertainty-aware shrinkage.

Reads submission.csv from each of N upstream kernels (top3, 9956, sani, pilkwang).
For each test row:
  - median of N predictions   (robust ensemble)
  - std across N predictions  (uncertainty)
  - last_known_tvt            (the safe value)

Output rule (regression analog of GeoHab's "flip only on consensus"):
  - if std <= LOW:    use median (high-confidence consensus)
  - if std >= HIGH:   shrink toward last_known_tvt   (no consensus → conservative)
  - in between:      linear blend median↔last_known_tvt weighted by std

Three regions instead of binary flip-or-not, because regression has no
'wrong category to flip away from' — the safe move is to shrink toward the
prior. This is the equivalent of GeoHab's "don't override the GBDT argmax
unless multiple models AND spatial NN agree".
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----- locate comp data + upstream submissions -----
COMP_ROOT = None
for p in Path("/kaggle/input").rglob("sample_submission.csv"):
    COMP_ROOT = p.parent
    break
if COMP_ROOT is None:
    sys.exit("error: comp data not mounted")
print(f"comp root: {COMP_ROOT}", flush=True)

# Find each upstream submission.csv
UPSTREAM_KEYS = ["top3", "9956", "sani", "pilkwang"]
upstream_paths = {}
for p in Path("/kaggle/input").rglob("submission.csv"):
    if "competitions/" in str(p):
        continue  # skip comp's own sample_submission
    for k in UPSTREAM_KEYS:
        if k in str(p).lower():
            upstream_paths[k] = p
            break

print(f"=== upstream submissions found: {len(upstream_paths)} ===", flush=True)
for k, p in upstream_paths.items():
    print(f"  {k}: {p}  ({os.path.getsize(p)} bytes)", flush=True)

if len(upstream_paths) < 2:
    sys.exit("need at least 2 upstream submissions to blend")

# ----- load all into a single dataframe -----
dfs = []
for k, p in upstream_paths.items():
    d = pd.read_csv(p).rename(columns={"tvt": f"tvt_{k}"})
    dfs.append(d)
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="id", how="inner")
print(f"merged: {merged.shape}", flush=True)
print(merged.head(3).to_string(index=False), flush=True)

pred_cols = [c for c in merged.columns if c.startswith("tvt_")]
print(f"\nN models: {len(pred_cols)}: {pred_cols}", flush=True)
print("\nstats per model:", flush=True)
print(merged[pred_cols].describe().T[["mean", "std", "min", "max"]].to_string(), flush=True)

# Pairwise disagreement matrix
print("\npairwise mean |diff| (ft):", flush=True)
preds = merged[pred_cols].values
for i in range(len(pred_cols)):
    for j in range(i + 1, len(pred_cols)):
        d = float(np.mean(np.abs(preds[:, i] - preds[:, j])))
        print(f"  {pred_cols[i]:14s} vs {pred_cols[j]:14s}: {d:.3f}", flush=True)

# ----- compute per-row last_known_tvt from test data -----
test_dir = COMP_ROOT / "test"
last_known = {}
for hp in sorted(test_dir.glob("*horizontal_well.csv")):
    well = hp.name.split("__")[0]
    h = pd.read_csv(hp)
    nan_mask = h["TVT_input"].isna()
    if not nan_mask.any():
        continue
    ps = int(nan_mask.idxmax())
    lk = float(h["TVT_input"].iloc[ps - 1])
    for i in range(ps, len(h)):
        last_known[f"{well}_{i}"] = lk
merged["last_known"] = merged["id"].map(last_known)
print(f"\nlast_known filled: {merged['last_known'].notna().sum()}/{len(merged)}", flush=True)

# ----- consensus + shrinkage -----
preds = merged[pred_cols].values
merged["median"] = np.median(preds, axis=1)
merged["std"] = np.std(preds, axis=1, ddof=0)
merged["mean_pred"] = np.mean(preds, axis=1)

print(f"\nstd distribution (across models per row):", flush=True)
print(merged["std"].describe().to_string(), flush=True)

LOW = 1.0    # ft — std below this = high confidence, use median directly
HIGH = 5.0   # ft — std above this = no consensus, full shrink to last_known

def shrink(std, low=LOW, high=HIGH):
    """Returns weight of `last_known` in the blend (0 = use median, 1 = use last_known)."""
    return np.clip((std - low) / max(high - low, 1e-6), 0.0, 1.0)

w_lk = shrink(merged["std"].values)
merged["tvt"] = (1 - w_lk) * merged["median"] + w_lk * merged["last_known"]

print(f"\nshrinkage weights distribution:", flush=True)
print(pd.Series(w_lk).describe().to_string(), flush=True)
print(f"\nrows fully shrunk to last_known: {int((w_lk >= 1.0).sum())} / {len(merged)}", flush=True)
print(f"rows trusting median fully:      {int((w_lk <= 0.0).sum())} / {len(merged)}", flush=True)

# ----- build submission in sample_submission order -----
sub = pd.read_csv(COMP_ROOT / "sample_submission.csv")
sub["tvt"] = sub["id"].map(merged.set_index("id")["tvt"])
n_missing = sub["tvt"].isna().sum()
if n_missing > 0:
    fallback = merged.set_index("id")["median"]
    print(f"WARN: {n_missing} missing → filling with median", flush=True)
    sub["tvt"] = sub["tvt"].fillna(sub["id"].map(fallback))
    sub["tvt"] = sub["tvt"].fillna(merged["mean_pred"].median())

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: shape={sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
