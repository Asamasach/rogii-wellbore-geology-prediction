"""LGB v3 — adds formation plane-fit (spatial KNN) features.

For each train well, extract a centroid: (X_med, Y_med, ANCC_med, ASTNU_med,
ASTNL_med, EGFDU_med, EGFDL_med, BUDA_med). 773 centroids total.

For each post-PS row in train OR test:
  - Query K=10 nearest centroids by (X, Y)
  - IDW-weighted (1/d^2) average of each formation Z -> 6 estimated formation depths
  - For TRAIN: leave-this-well-out (don't use the row's own well centroid)
  - For TEST: use all 773 train centroids

Then add per-row features:
  - ancc_est, astnu_est, astnl_est, egfdu_est, egfdl_est, buda_est
  - z_to_<form> = h.Z - form_est (signed distance from current wellbore Z to
    formation top — strong geological signal)
  - lk_to_<form> = last_known_tvt - form_est (where the known TVT sits relative
    to each formation top)

12 new features in addition to v2's 44 -> 56 total.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd

from predict_tvt_lgb_v2 import build_features_for_well_v2

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def build_centroids(train_dir: Path) -> pd.DataFrame:
    """Per train well: (well, X, Y, ancc_z, astnu_z, ..., buda_z) -- median over rows."""
    rows = []
    for hp in sorted(train_dir.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        h = pd.read_csv(hp)
        if not all(f in h.columns for f in FORMATIONS):
            # Not a labelled-train well (e.g. test horizontal has no formation cols)
            continue
        rec = {"well": well, "X": float(h["X"].median()), "Y": float(h["Y"].median())}
        for f in FORMATIONS:
            rec[f"{f.lower()}_z"] = float(h[f].median())
        rows.append(rec)
    return pd.DataFrame(rows)


def estimate_formations_for_xy(
    xy_query: np.ndarray,         # (n_query, 2)
    centroids: pd.DataFrame,
    exclude_well: str | None = None,
    k: int = 10,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """K-nearest IDW interpolation of each formation Z at the query (X,Y) points.

    Returns dict { 'ancc': arr, 'astnu': arr, ... } shape (n_query,).
    """
    if exclude_well is not None:
        c = centroids[centroids["well"] != exclude_well].reset_index(drop=True)
    else:
        c = centroids
    cxy = c[["X", "Y"]].values
    out = {f.lower(): np.full(len(xy_query), np.nan, dtype=np.float64) for f in FORMATIONS}
    if len(c) == 0:
        return out

    # Distances (n_query, n_centroids)
    # Vectorise — sufficient for n_centroids ~770 and n_query ~5000 per well
    diff = xy_query[:, None, :] - cxy[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    # Get K nearest indices per query
    K = min(k, d2.shape[1])
    # argpartition for top-K (faster than argsort for whole row)
    idx = np.argpartition(d2, K - 1, axis=1)[:, :K]
    # Gather distances
    rows = np.arange(len(xy_query))[:, None]
    nn_d2 = d2[rows, idx]
    w = 1.0 / (nn_d2 + eps)
    w_sum = w.sum(axis=1, keepdims=True)
    for f in FORMATIONS:
        col = f"{f.lower()}_z"
        vals = c[col].values[idx]  # (n_query, K)
        out[f.lower()] = (vals * w).sum(axis=1) / w_sum.flatten()
    return out


def add_formation_features(df: pd.DataFrame, centroids: pd.DataFrame, well: str,
                           is_train: bool):
    """Add the 12 formation-related features in place."""
    xy = df[["X", "Y"]].values
    exclude = well if is_train else None
    est = estimate_formations_for_xy(xy, centroids, exclude_well=exclude, k=10)
    z_arr = df["Z"].to_numpy()
    lk_arr = df["last_known_tvt"].to_numpy()
    for f in FORMATIONS:
        key = f.lower()
        df[f"{key}_est"] = est[key]
        df[f"z_to_{key}"] = z_arr - est[key]
        # last_known_tvt - formation top — in TVT space
        # Note: TVT and Z are in DIFFERENT spaces (TVT is depth-along-typewell, Z is
        # cartesian Z of horizontal). They aren't directly comparable, but the
        # difference is still a useful relative position signal.
        df[f"lk_to_{key}"] = lk_arr - est[key]


def build_features_for_well_v3(h: pd.DataFrame, t: pd.DataFrame, well: str,
                               centroids: pd.DataFrame, is_train: bool,
                               include_target: bool = True) -> pd.DataFrame:
    df = build_features_for_well_v2(h, t, well, include_target=include_target)
    if len(df) == 0:
        return df
    add_formation_features(df, centroids, well, is_train)
    return df


def build_dataset_v3(train_dir: Path, centroids: pd.DataFrame, is_train: bool,
                     include_target: bool = True):
    parts = []
    files = sorted(train_dir.glob("*horizontal_well.csv"))
    for i, hp in enumerate(files):
        well = hp.name.split("__")[0]
        tp = train_dir / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well_v3(h, t, well, centroids, is_train, include_target)
        if len(df):
            parts.append(df)
        if (i + 1) % 100 == 0:
            print(f"  built v3 features {i+1}/{len(files)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----- driver: 4-sample local test (no test data, just verify shape) -----
if __name__ == "__main__":
    SAMPLES = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_train")
    print("=== build centroids from 4 sample wells ===")
    centroids = build_centroids(SAMPLES)
    print(centroids.to_string(index=False))

    print("\n=== build v3 features for 4 sample wells (LOWO) ===")
    parts = []
    for hp in sorted(SAMPLES.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        tp = SAMPLES / f"{well}__typewell.csv"
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well_v3(h, t, well, centroids, is_train=True)
        parts.append(df)
        print(f"  {well}: {df.shape}")
    full = pd.concat(parts, ignore_index=True)
    new_cols = [c for c in full.columns
                if c.endswith("_est") or c.startswith("z_to_") or c.startswith("lk_to_")]
    print(f"\nformation features ({len(new_cols)}): {new_cols}")
    print(f"sample stats:\n{full[new_cols].describe().T[['mean','std','min','max']]}")

    if "target" in full.columns:
        print("\ncorrelations with target:")
        for col in new_cols:
            c = float(full[col].corr(full["target"]))
            print(f"  {col:20s}: {c:+.4f}")
