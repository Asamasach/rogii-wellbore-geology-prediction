"""Generate sub_const_v1.csv from the 3 test wells using the const baseline.

Reads test horizontal wells from fteam6 (or local samples), computes
TVT[i>=PS] = TVT_input[PS-1], and writes a submission CSV in the format
matching sample_submission.csv: id = '<well_id>_<row_index>', tvt = predicted.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import numpy as np

# Detect environment: prefer remote test/ on fteam6 if present, else use local samples
ROOT_REMOTE = Path("/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw/test")
ROOT_LOCAL = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_test")
if ROOT_REMOTE.exists():
    ROOT = ROOT_REMOTE
    OUT = Path("/home/fteam6/project/rogii-wellbore-geology-prediction/submissions/sub_const_v1.csv")
else:
    ROOT = ROOT_LOCAL
    OUT = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\submissions\sub_const_v1.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"reading from {ROOT}")
test_h_files = sorted(ROOT.glob("*horizontal_well.csv"))
print(f"found {len(test_h_files)} horizontal test wells")

rows = []
for hp in test_h_files:
    well_id = hp.name.split("__")[0]
    h = pd.read_csv(hp)
    nan_mask = h["TVT_input"].isna()
    if not nan_mask.any():
        print(f"  {well_id}: no NaN in TVT_input — skipping")
        continue
    ps = int(nan_mask.idxmax())
    n_pred = len(h) - ps
    last_known = float(h["TVT_input"].iloc[ps - 1])
    print(f"  {well_id}: rows={len(h)} PS={ps} predict={n_pred} last_TVT={last_known:.2f}")
    for i in range(ps, len(h)):
        # Row indices in sample_submission start at PS for each well
        rows.append({"id": f"{well_id}_{i}", "tvt": last_known})

sub = pd.DataFrame(rows)
sub.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
print(f"  shape: {sub.shape}")
print(f"  head:\n{sub.head().to_string(index=False)}")
print(f"  tail:\n{sub.tail().to_string(index=False)}")
