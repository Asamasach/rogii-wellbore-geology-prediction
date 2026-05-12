"""ROGII v10 — weighted blend of 5 forks. Favours ultra (claimed sub-9) heaviest."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {
    "ultra":    0.35,  # sub-9 claim
    "tasmim":   0.20,  # 9.830 claim
    "top3":     0.20,  # claimed top-3 (~9.4)
    "9956":     0.15,  # confirmed 10.076
    "pilkwang": 0.10,  # ~10 expected
}

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

print(f"upstreams: {len(upstream_paths)}", flush=True)
for k, p in upstream_paths.items():
    print(f"  {k}: {p}", flush=True)
if len(upstream_paths) < 2:
    sys.exit("need >= 2")

present_w_sum = sum(WEIGHTS[k] for k in upstream_paths)
norm_w = {k: WEIGHTS[k] / present_w_sum for k in upstream_paths}
print(f"normalised weights:")
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
