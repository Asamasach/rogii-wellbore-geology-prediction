"""LGB v2 — adds NCC alignment features to v1.

For each post-PS row i, compute a small set of "best-match-typewell-TVT"
estimates by sliding a horizontal-GR window over a typewell-GR neighbourhood
of last_known_tvt (within ±SEARCH_RADIUS feet). For each candidate typewell
position j, score = mean((hw_window - tw_window)^2). Track the BEST and a
few other statistics:

    ncc_pred_tvt_w<W> : typewell TVT at the best match for window W
    ncc_pred_dlt_w<W> : ncc_pred_tvt - last_known_tvt
    ncc_score_w<W>    : best score (lower = better match)
    ncc_score_w<W>_at_lk : score at last_known_tvt position
    ncc_score_ratio_w<W> : ratio of best score to score-at-lk (1 = same; <1 = better elsewhere)

Three window sizes: 11, 25, 51 -> 5 features each = 15 new features.

Search is vectorised per-row using np.lib.stride_tricks for the typewell
window batch + broadcasting against the (single) horizontal window.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd

from predict_tvt import detect_ps, fillna_gr
from predict_tvt_lgb import build_features_for_well as v1_build

NCC_WINDOWS = (11, 25, 51)
SEARCH_RADIUS_FT = 30  # typewell positions within last_known_tvt ± this


def _interp(t_tvt, t_gr, x):
    return np.interp(x, t_tvt, t_gr)


def _sliding_windows(arr, win):
    """Return shape (n - win + 1, win) view of `arr` (1D)."""
    if len(arr) < win:
        return np.empty((0, win), dtype=arr.dtype)
    return np.lib.stride_tricks.sliding_window_view(arr, win)


def add_ncc_features(df: pd.DataFrame, h: pd.DataFrame, t: pd.DataFrame, ps: int):
    """Mutate df in place: add NCC_* columns."""
    h_gr_full = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy()
    t_gr = fillna_gr(t["GR"].to_numpy())
    if not np.all(np.diff(t_tvt) >= 0):
        order = np.argsort(t_tvt); t_tvt = t_tvt[order]; t_gr = t_gr[order]

    last_known_tvt = float(h["TVT_input"].iloc[ps - 1])
    n_post = len(df)

    # Typewell candidate window: indices whose TVT is within last_known_tvt ± R
    lo_tvt = last_known_tvt - SEARCH_RADIUS_FT
    hi_tvt = last_known_tvt + SEARCH_RADIUS_FT
    cand_lo = int(np.searchsorted(t_tvt, lo_tvt, side="left"))
    cand_hi = int(np.searchsorted(t_tvt, hi_tvt, side="right"))
    if cand_hi - cand_lo < max(NCC_WINDOWS):
        # Fallback: use a wider window
        cand_lo = max(0, cand_lo - max(NCC_WINDOWS))
        cand_hi = min(len(t_tvt), cand_hi + max(NCC_WINDOWS))

    t_gr_cand_region = t_gr[cand_lo:cand_hi]
    t_tvt_cand_region = t_tvt[cand_lo:cand_hi]

    # Index of last_known_tvt within the candidate region, for "score-at-lk"
    lk_pos_in_region = int(np.clip(
        np.searchsorted(t_tvt_cand_region, last_known_tvt), 0, len(t_tvt_cand_region) - 1
    ))

    for W in NCC_WINDOWS:
        half = W // 2
        # Sliding windows over typewell candidate region
        tw_wins = _sliding_windows(t_gr_cand_region, W)  # (n_cand - W + 1, W)
        # The TVT for each typewell-window is the TVT at its centre
        tw_centre_tvt = t_tvt_cand_region[half : half + tw_wins.shape[0]]
        if tw_wins.shape[0] == 0:
            df[f"ncc_pred_tvt_w{W}"] = last_known_tvt
            df[f"ncc_pred_dlt_w{W}"] = 0.0
            df[f"ncc_score_w{W}"] = np.nan
            df[f"ncc_score_w{W}_at_lk"] = np.nan
            df[f"ncc_score_ratio_w{W}"] = 1.0
            continue

        # Per-row horizontal window
        # Horizontal index for row i in df is df["row_idx"].iloc[i]
        h_indices = df["row_idx"].to_numpy()
        h_gr_padded = h_gr_full  # use raw, no padding — clip windows at edges

        ncc_pred_tvt = np.full(n_post, last_known_tvt, dtype=np.float64)
        ncc_score = np.full(n_post, np.nan, dtype=np.float64)
        ncc_score_at_lk = np.full(n_post, np.nan, dtype=np.float64)

        # Find a "lk window" once per W (it's the typewell window centred at lk_pos_in_region)
        # Mapped to tw_wins index: lk_pos_in_region - half
        lk_win_idx = lk_pos_in_region - half
        if 0 <= lk_win_idx < tw_wins.shape[0]:
            lk_tw_win = tw_wins[lk_win_idx]
        else:
            lk_tw_win = tw_wins[tw_wins.shape[0] // 2]

        for ii, hi_global in enumerate(h_indices):
            lo = hi_global - half
            hi = hi_global + half + 1
            if lo < 0 or hi > len(h_gr_padded):
                # edge case: pad by mirroring/clipping
                continue
            h_win = h_gr_padded[lo:hi]
            if not np.isfinite(h_win).all():
                continue
            # Broadcast: tw_wins shape (M, W), h_win shape (W,)
            d = np.mean((tw_wins - h_win) ** 2, axis=1)
            best = int(np.argmin(d))
            ncc_pred_tvt[ii] = tw_centre_tvt[best]
            ncc_score[ii] = d[best]
            ncc_score_at_lk[ii] = float(np.mean((lk_tw_win - h_win) ** 2))

        df[f"ncc_pred_tvt_w{W}"] = ncc_pred_tvt
        df[f"ncc_pred_dlt_w{W}"] = ncc_pred_tvt - last_known_tvt
        df[f"ncc_score_w{W}"] = ncc_score
        df[f"ncc_score_w{W}_at_lk"] = ncc_score_at_lk
        df[f"ncc_score_ratio_w{W}"] = ncc_score / np.where(ncc_score_at_lk > 0, ncc_score_at_lk, 1.0)


def build_features_for_well_v2(h: pd.DataFrame, t: pd.DataFrame, well: str,
                               include_target: bool = True) -> pd.DataFrame:
    df = v1_build(h, t, well, include_target=include_target)
    if len(df) == 0:
        return df
    ps = detect_ps(h)
    add_ncc_features(df, h, t, ps)
    return df


def build_dataset_v2(train_dir: Path, max_wells: int | None = None,
                     include_target: bool = True):
    files = sorted(train_dir.glob("*horizontal_well.csv"))
    if max_wells:
        files = files[:max_wells]
    parts = []
    for i, hp in enumerate(files):
        well = hp.name.split("__")[0]
        tp = train_dir / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well_v2(h, t, well, include_target=include_target)
        if len(df):
            parts.append(df)
        if (i + 1) % 100 == 0:
            print(f"  built v2 features {i+1}/{len(files)}", flush=True)
    return pd.concat(parts, ignore_index=True)


# ----- driver: 4-sample test -----
if __name__ == "__main__":
    from predict_tvt import post_ps_rmse
    import time
    SAMPLES = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_train")
    parts = []
    t0 = time.time()
    for hp in sorted(SAMPLES.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        tp = SAMPLES / f"{well}__typewell.csv"
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        df = build_features_for_well_v2(h, t, well, include_target=True)
        parts.append(df)
        print(f"  {well}: {df.shape}  t={time.time()-t0:.1f}s")
    full = pd.concat(parts, ignore_index=True)
    print(f"\ntotal: {len(full)} rows, {len(full.columns)} cols")
    new_cols = [c for c in full.columns if c.startswith("ncc_")]
    print(f"NCC cols: {new_cols}")
    print(f"\nNCC stats:\n{full[new_cols].describe().T[['mean','std','min','max']]}")

    # Quick correlation: ncc_pred_dlt_w25 vs target
    if "target" in full.columns:
        for col in ["ncc_pred_dlt_w11", "ncc_pred_dlt_w25", "ncc_pred_dlt_w51",
                    "gr_diff", "last_dtvt"]:
            if col in full.columns:
                c = float(full[col].corr(full["target"]))
                print(f"  corr({col:25s}, target) = {c:+.4f}")
