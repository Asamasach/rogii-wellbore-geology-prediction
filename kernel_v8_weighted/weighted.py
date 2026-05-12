"""ROGII v8 — weighted blend favouring strongest model (NO shrinkage).

Weights driven by expected LB tier (lower expected RMSE = higher weight):
  top3      0.50  (claimed LB ~9.4)
  9956      0.30  (confirmed LB 10.076)
  pilkwang  0.20  (LB ~10 expected)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {"top3": 0.50, "9956": 0.30, "pilkwang": 0.20}

COMP_ROOT = None
for p in Path("/kaggle/input").rglob("sample_submission.csv"):
    COMP_ROOT = p.parent
    break
if COMP_ROOT is None:
    sys.exit("error: comp data not mounted")

upstream_paths = {}
for p in Path("/kaggle/input").rglob("submission.csv"):
    if "competitions/" in str(p):
        continue
    for k in WEIGHTS:
        if k in str(p).lower():
            upstream_paths[k] = p
            break

print(f"upstream submissions: {len(upstream_paths)}", flush=True)
for k, p in upstream_paths.items():
    print(f"  {k}: {p}", flush=True)

if len(upstream_paths) < 2:
    sys.exit("need at least 2 upstreams")

# Normalise weights over present subset
present_w_sum = sum(WEIGHTS[k] for k in upstream_paths)
norm_w = {k: WEIGHTS[k] / present_w_sum for k in upstream_paths}
print(f"\nnormalised weights:", flush=True)
for k, w in norm_w.items():
    print(f"  {k}: {w:.3f}", flush=True)

dfs = []
for k, p in upstream_paths.items():
    d = pd.read_csv(p).rename(columns={"tvt": f"tvt_{k}"})
    dfs.append(d)
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="id", how="inner")

merged["tvt"] = 0.0
for k, w in norm_w.items():
    merged["tvt"] += w * merged[f"tvt_{k}"]

sub = pd.read_csv(COMP_ROOT / "sample_submission.csv")
sub["tvt"] = sub["id"].map(merged.set_index("id")["tvt"])
sub["tvt"] = sub["tvt"].fillna(merged[f"tvt_{list(WEIGHTS)[0]}"].median())

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: {sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
