"""ROGII v5 — blend submissions from two upstream kernels:
  - asamasach/rogii-better-solution-9956 (LB ~9.956 expected)
  - asamasach/rogii-lgb-v4 (LB ~13-14 expected)

Default weight: 0.85 * 9956 + 0.15 * v4 (9956 dominant; v4 adds diversity).
"""
import os
import sys
from pathlib import Path

import pandas as pd

W_9956 = 0.85
W_V4 = 1.0 - W_9956

# Find the input submission files. Kaggle mounts kernel outputs at
# /kaggle/input/<kernel-slug>/<file>
ROOT = Path("/kaggle/input")
print(f"=== /kaggle/input contents ===", flush=True)
for p in sorted(ROOT.iterdir()):
    print(f"  {p}", flush=True)
    if p.is_dir():
        for sub in sorted(p.iterdir())[:5]:
            print(f"    {sub.name}", flush=True)

# Auto-find submission.csv from each upstream kernel
candidates = list(ROOT.rglob("submission.csv"))
print(f"\nfound {len(candidates)} submission.csv files:", flush=True)
for c in candidates:
    print(f"  {c}", flush=True)

# Pick by directory name
sub_9956_path = None
sub_v4_path = None
for c in candidates:
    if "9956" in str(c).lower() or "better-solution" in str(c).lower():
        sub_9956_path = c
    elif "v4" in str(c).lower() or "lgb-v4" in str(c).lower():
        sub_v4_path = c

if sub_9956_path is None or sub_v4_path is None:
    print(f"WARN: missing one of the upstream submissions. v9956={sub_9956_path}, v4={sub_v4_path}",
          flush=True)
    if sub_9956_path is None and sub_v4_path is None:
        sys.exit("no submissions to blend")
    # Fallback: use whichever is present
    only = sub_9956_path or sub_v4_path
    sub = pd.read_csv(only)
    print(f"using only {only}: shape={sub.shape}", flush=True)
else:
    sub_9956 = pd.read_csv(sub_9956_path)
    sub_v4 = pd.read_csv(sub_v4_path)
    print(f"\nv9956: {sub_9956.shape}  cols={list(sub_9956.columns)}", flush=True)
    print(f"v4:    {sub_v4.shape}  cols={list(sub_v4.columns)}", flush=True)

    merged = sub_9956.rename(columns={"tvt": "tvt_9956"}).merge(
        sub_v4.rename(columns={"tvt": "tvt_v4"}), on="id", how="inner"
    )
    print(f"merged: {merged.shape}", flush=True)
    diff_stats = (merged["tvt_9956"] - merged["tvt_v4"]).describe()
    print(f"\n9956 - v4 stats:\n{diff_stats}", flush=True)

    merged["tvt"] = W_9956 * merged["tvt_9956"] + W_V4 * merged["tvt_v4"]
    sub = merged[["id", "tvt"]]

OUT = "/kaggle/working/submission.csv"
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}: shape={sub.shape}", flush=True)
print(f"tvt stats: mean={sub['tvt'].mean():.2f} std={sub['tvt'].std():.2f}", flush=True)
print(sub.head().to_string(index=False), flush=True)
