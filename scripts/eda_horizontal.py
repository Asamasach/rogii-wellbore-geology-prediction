"""EDA on the horizontal_well + matched typewell pairs (4 train, 3 test)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd

ROOT = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw")

print("=" * 70)
print("TRAIN horizontal_wells (4 sampled)")
print("=" * 70)
for h in sorted((ROOT / "_samples_train").glob("*horizontal_well.csv")):
    df = pd.read_csv(h)
    well_id = h.name.split("__")[0]
    print(f"\n{h.name}")
    print(f"  shape: {df.shape}")
    print(f"  cols : {list(df.columns)}")
    # Identify Prediction Start (PS) — first row where TVT_input is NaN but TVT is known
    has_tvt = "TVT" in df.columns
    has_input = "TVT_input" in df.columns
    if has_tvt and has_input:
        ps_idx = df["TVT_input"].isna().idxmax() if df["TVT_input"].isna().any() else len(df)
        print(f"  PS row index: {ps_idx} (out of {len(df)}) -> {len(df) - ps_idx} prediction points")
        print(f"  TVT (full)      range: [{df['TVT'].min():.2f}, {df['TVT'].max():.2f}]  NaN={df['TVT'].isna().sum()}")
        print(f"  TVT_input range:        [{df['TVT_input'].min():.2f}, {df['TVT_input'].max():.2f}]  NaN={df['TVT_input'].isna().sum()}")
    if "MD" in df.columns:
        print(f"  MD range: [{df['MD'].min():.2f}, {df['MD'].max():.2f}]  step ~{(df['MD'].diff().median()):.2f}")
    if "GR" in df.columns:
        print(f"  GR range: [{df['GR'].min():.2f}, {df['GR'].max():.2f}]  NaN={df['GR'].isna().sum()}")
    if {"X", "Y", "Z"}.issubset(df.columns):
        print(f"  X range: [{df['X'].min():.0f}, {df['X'].max():.0f}]  Δ={df['X'].max()-df['X'].min():.0f}")
        print(f"  Y range: [{df['Y'].min():.0f}, {df['Y'].max():.0f}]  Δ={df['Y'].max()-df['Y'].min():.0f}")
        print(f"  Z range: [{df['Z'].min():.0f}, {df['Z'].max():.0f}]  Δ={df['Z'].max()-df['Z'].min():.0f}")
    print(f"  head:\n{df.head(3).to_string(index=False)}")

print()
print("=" * 70)
print("TEST horizontal_wells (3 — full set)")
print("=" * 70)
for h in sorted((ROOT / "_samples_test").glob("*horizontal_well.csv")):
    df = pd.read_csv(h)
    print(f"\n{h.name}")
    print(f"  shape: {df.shape}")
    print(f"  cols : {list(df.columns)}")
    if "TVT_input" in df.columns:
        ps_idx = df["TVT_input"].isna().idxmax() if df["TVT_input"].isna().any() else len(df)
        print(f"  PS row index: {ps_idx} -> {len(df) - ps_idx} points need prediction")
        print(f"  TVT_input NaN={df['TVT_input'].isna().sum()}")
    print(f"  head:\n{df.head(3).to_string(index=False)}")
