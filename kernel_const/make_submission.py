"""ROGII const baseline submission: TVT[i>=PS] = TVT_input[PS-1] for each test well.

Auto-detects the competition input mount path and writes /kaggle/working/submission.csv.
"""
import os
import sys
from pathlib import Path

import pandas as pd

# Find comp data root
candidates = []
for p in Path("/kaggle/input").rglob("*"):
    if p.is_file() and p.name.endswith("__horizontal_well.csv") and "test" in str(p):
        candidates.append(p.parent)
test_dir = sorted(set(candidates))[0]
comp_root = test_dir.parent
print(f"comp root: {comp_root}")
print(f"test dir : {test_dir}")

test_h_files = sorted(test_dir.glob("*horizontal_well.csv"))
print(f"test horizontal wells: {len(test_h_files)}")

rows = []
for hp in test_h_files:
    well = hp.name.split("__")[0]
    h = pd.read_csv(hp)
    nan_mask = h["TVT_input"].isna()
    if not nan_mask.any():
        continue
    ps = int(nan_mask.idxmax())
    last_known = float(h["TVT_input"].iloc[ps - 1])
    print(f"  {well}: rows={len(h)} PS={ps} predict={len(h)-ps} last={last_known:.2f}")
    for i in range(ps, len(h)):
        rows.append({"id": f"{well}_{i}", "tvt": last_known})

sub = pd.DataFrame(rows)
out = "/kaggle/working/submission.csv"
sub.to_csv(out, index=False)
print(f"\nwrote {out}: {sub.shape}")
print(sub.head().to_string(index=False))
