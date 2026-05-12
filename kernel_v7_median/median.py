"""ROGII v7 — simple median of 3 strong forks (top3 + 9956 + pilkwang).

CORRECT lesson from v6 disaster: for regression on this comp,
  - DO take median across STRONG forks (robust to outlier)
  - DO NOT shrink toward last_known_tvt (kills predictions where well actually drifted)

Just median across 3 forks. That's the whole script.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# locate comp data + upstream subs
COMP_ROOT = None
for p in Path("/kaggle/input").rglob("sample_submission.csv"):
    COMP_ROOT = p.parent
    break
if COMP_ROOT is None:
    sys.exit("error: comp data not mounted")

UPSTREAM_KEYS = ["top3", "9956", "pilkwang"]
upstream_paths = {}
for p in Path("/kaggle/input").rglob("submission.csv"):
    if "competitions/" in str(p):
        continue
    for k in UPSTREAM_KEYS:
        if k in str(p).lower():
            upstream_paths[k] = p
            break

print(f"upstream submissions: {len(upstream_paths)}", flush=True)
for k, p in upstream_paths.items():
    print(f"  {k}: {p}  ({os.path.getsize(p)} bytes)", flush=True)

if len(upstream_paths) < 2:
    sys.exit("need at least 2 upstreams")

# Load + merge
dfs = []
for k, p in upstream_paths.items():
    d = pd.read_csv(p).rename(columns={"tvt": f"tvt_{k}"})
    dfs.append(d)
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="id", how="inner")
pred_cols = [c for c in merged.columns if c.startswith("tvt_")]

# Pairwise mean |diff|
print("\npairwise mean |diff| (ft):", flush=True)
preds = merged[pred_cols].values
for i in range(len(pred_cols)):
    for j in range(i + 1, len(pred_cols)):
        d = float(np.mean(np.abs(preds[:, i] - preds[:, j])))
        print(f"  {pred_cols[i]:14s} vs {pred_cols[j]:14s}: {d:.3f}", flush=True)

# Simple median — robust to one model being far off
merged["tvt"] = np.median(preds, axis=1)

# Build submission in sample_submission order
sub = pd.read_csv(COMP_ROOT / "sample_submission.csv")
sub["tvt"] = sub["id"].map(merged.set_index("id")["tvt"])
n_missing = sub["tvt"].isna().sum()
if n_missing > 0:
    # Fallback to mean if any rows missing
    fallback = merged.set_index("id")[pred_cols].mean(axis=1)
    sub["tvt"] = sub["tvt"].fillna(sub["id"].map(fallback))
    sub["tvt"] = sub["tvt"].fillna(merged[pred_cols[0]].median())

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: {sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
