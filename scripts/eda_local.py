"""Read the locally-pulled samples and characterise the data."""
import pandas as pd
from pathlib import Path

ROOT = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw")

print("=" * 70)
print("SAMPLE SUBMISSION")
print("=" * 70)
ss = pd.read_csv(ROOT / "sample_submission.csv")
print(f"shape : {ss.shape}")
print(f"cols  : {list(ss.columns)}")
print(f"dtypes:\n{ss.dtypes}")
print(f"head:\n{ss.head(8)}")
print(f"tail:\n{ss.tail(5)}")
# Are there multiple distinct well_ids?
ss["well_id"] = ss["id"].str.split("_").str[0]
print(f"\nunique well_ids in sample_submission: {ss['well_id'].nunique()}")
print(f"  {ss['well_id'].unique()}")
print(f"rows per well_id: {ss.groupby('well_id').size().to_dict()}")
print(f"tvt distribution (if non-zero):\n{ss['tvt'].describe()}")
# Inspect a few row IDs to see the indexing pattern within a well
for wid in ss["well_id"].unique()[:2]:
    sub = ss[ss["well_id"] == wid].head(5)
    print(f"\nfirst rows for {wid}:\n{sub[['id','tvt']]}")

print()
print("=" * 70)
print("TRAIN TYPEWELLS (8 samples)")
print("=" * 70)
train_dir = ROOT / "_samples_train"
train_files = sorted(train_dir.glob("*.csv"))
all_summaries = []
for f in train_files:
    df = pd.read_csv(f)
    geo_vals = df["Geology"].dropna().value_counts()
    all_summaries.append({
        "file": f.name,
        "rows": len(df),
        "tvt_min": df["TVT"].min(),
        "tvt_max": df["TVT"].max(),
        "gr_min": df["GR"].min(),
        "gr_max": df["GR"].max(),
        "gr_mean": df["GR"].mean(),
        "geo_unique": list(geo_vals.index),
        "geo_total_labels": int(geo_vals.sum()),
        "geo_pct_labeled": 100.0 * int(geo_vals.sum()) / len(df),
    })
summ = pd.DataFrame(all_summaries)
print(summ.to_string(index=False))

# Aggregate Geology label vocab across all train typewells
print()
all_geo = pd.concat([pd.read_csv(f)["Geology"].dropna() for f in train_files])
print(f"Geology label vocab (train typewells):")
print(all_geo.value_counts())
print(f"Total Geology labels: {len(all_geo)}")

print()
print("=" * 70)
print("TEST TYPEWELLS (3 wells)")
print("=" * 70)
test_dir = ROOT / "_samples_test"
test_files = sorted(test_dir.glob("*.csv"))
for f in test_files:
    df = pd.read_csv(f)
    print(f"\n{f.name}: {df.shape}")
    print(f"  TVT range: [{df['TVT'].min():.2f}, {df['TVT'].max():.2f}]")
    print(f"  GR range:  [{df['GR'].min():.2f}, {df['GR'].max():.2f}]")
    geo = df["Geology"].dropna()
    print(f"  Geology labels present: {len(geo)} ({100*len(geo)/len(df):.1f}%)")
    if len(geo):
        print(f"    vocab: {geo.value_counts().to_dict()}")
    print(f"  head:\n{df.head(3).to_string(index=False)}")
