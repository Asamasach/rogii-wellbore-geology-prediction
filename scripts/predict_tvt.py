"""TVT prediction baselines for ROGII Wellbore Geology Prediction.

Three predictors evaluated post-PS:
  1. CONST   : TVT[i>=PS] = TVT_input[PS-1]              (lower bound)
  2. LINEAR  : extrapolate TVT_input slope w.r.t. MD     (decent if ~horizontal)
  3. ALIGN   : local GR-window alignment to typewell     (the physical model)

Each predictor takes (horizontal_df, typewell_df) and returns a 1D array of
predicted TVT for every horizontal row. Evaluation is RMSE on rows >= PS where
horizontal['TVT'] is known (training mode only).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- helpers --------------------------------------

def detect_ps(h: pd.DataFrame) -> int:
    """Return index of first NaN in TVT_input — the Prediction Start point.
    If TVT_input has no NaNs, PS is len(h) (no rows to predict)."""
    nan_mask = h["TVT_input"].isna()
    return int(nan_mask.idxmax()) if nan_mask.any() else len(h)


def fillna_gr(arr):
    """Linear-interpolate NaN GR values; ffill/bfill ends."""
    s = pd.Series(arr)
    return s.interpolate(limit_direction="both").to_numpy()


# ---------------------------- predictors ------------------------------------

def predict_const(h: pd.DataFrame, t: pd.DataFrame, ps: int) -> np.ndarray:
    """Hold last known TVT for all i >= PS."""
    out = h["TVT_input"].to_numpy().copy()
    last = h["TVT_input"].iloc[ps - 1] if ps > 0 else 0.0
    out[ps:] = last
    return out


def predict_linear(h: pd.DataFrame, t: pd.DataFrame, ps: int, win: int = 100) -> np.ndarray:
    """Fit a linear TVT~MD on the last `win` calibration rows, extrapolate forward."""
    out = h["TVT_input"].to_numpy().copy()
    if ps == 0:
        return out
    s = max(0, ps - win)
    md = h["MD"].iloc[s:ps].to_numpy()
    tv = h["TVT_input"].iloc[s:ps].to_numpy()
    # least-squares slope
    if len(md) >= 2 and np.ptp(md) > 0:
        a, b = np.polyfit(md, tv, 1)
    else:
        a, b = 0.0, h["TVT_input"].iloc[ps - 1]
    md_post = h["MD"].iloc[ps:].to_numpy()
    out[ps:] = a * md_post + b
    return out


def predict_align(
    h: pd.DataFrame, t: pd.DataFrame, ps: int,
    gr_win: int = 25,
    max_step: float = 1.5,
    z_normalize: bool = True,
) -> np.ndarray:
    """Local GR-window alignment to typewell, propagating from PS.

    For each i >= PS:
      - extract GR window of size `gr_win` around i in horizontal
      - search typewell points whose TVT is within ±max_step of last_pred_TVT
        (limits how fast TVT can change between consecutive 1-ft MD steps)
      - score each candidate by L2 distance between horizontal GR window and
        typewell GR window centred at the candidate's TVT
      - pick the best, set last_pred_TVT = candidate.TVT

    Initialised from TVT_input[PS-1].
    """
    out = h["TVT_input"].to_numpy().copy().astype(np.float64)
    if ps == 0:
        return out
    if ps >= len(h):
        return out

    h_gr = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy()
    t_gr = fillna_gr(t["GR"].to_numpy())

    # Pre-normalise per-well for scale invariance
    if z_normalize:
        h_gr_n = (h_gr - np.nanmean(h_gr)) / (np.nanstd(h_gr) + 1e-6)
        t_gr_n = (t_gr - np.nanmean(t_gr)) / (np.nanstd(t_gr) + 1e-6)
    else:
        h_gr_n, t_gr_n = h_gr, t_gr

    half = gr_win // 2
    n_h = len(h)
    n_t = len(t)

    # Precompute typewell windowed-GR for fast lookup: just slice on demand
    # because n_t is ~1k-2k.

    last_tvt = float(out[ps - 1])
    # also: which typewell index has TVT ≈ last_tvt at start?
    last_t_idx = int(np.searchsorted(t_tvt, last_tvt))

    for i in range(ps, n_h):
        lo_h = max(0, i - half)
        hi_h = min(n_h, i + half + 1)
        win_h = h_gr_n[lo_h:hi_h]
        # candidate typewell positions: TVT within [last_tvt - max_step, last_tvt + max_step]
        # Both directions allowed (drilling can steer up or down)
        lo_tvt = last_tvt - max_step
        hi_tvt = last_tvt + max_step
        lo_idx = int(np.searchsorted(t_tvt, lo_tvt, side="left"))
        hi_idx = int(np.searchsorted(t_tvt, hi_tvt, side="right"))
        # Ensure at least 1 candidate
        if hi_idx <= lo_idx:
            lo_idx = max(0, last_t_idx - 1)
            hi_idx = min(n_t, last_t_idx + 2)

        best_j = last_t_idx
        best_score = float("inf")
        for j in range(lo_idx, hi_idx):
            jl = max(0, j - half)
            jr = min(n_t, j + half + 1)
            win_t = t_gr_n[jl:jr]
            # Match window lengths (clip on short ends)
            L = min(len(win_h), len(win_t))
            if L < 3:
                continue
            d = np.mean((win_h[:L] - win_t[:L]) ** 2)
            if d < best_score:
                best_score = d
                best_j = j

        last_tvt = float(t_tvt[best_j])
        last_t_idx = best_j
        out[i] = last_tvt

    return out


# --------------------------- evaluation -------------------------------------

def post_ps_rmse(true_tvt: np.ndarray, pred_tvt: np.ndarray, ps: int) -> float:
    if ps >= len(true_tvt):
        return float("nan")
    err = true_tvt[ps:] - pred_tvt[ps:]
    return float(np.sqrt(np.mean(err ** 2)))


PREDICTORS = {
    "const": predict_const,
    "linear": predict_linear,
    "align": predict_align,
}


# --------------------------- driver -----------------------------------------

if __name__ == "__main__":
    SAMPLES_DIR = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_train")
    rows = []
    for h_path in sorted(SAMPLES_DIR.glob("*horizontal_well.csv")):
        well = h_path.name.split("__")[0]
        t_path = SAMPLES_DIR / f"{well}__typewell.csv"
        if not t_path.exists():
            print(f"  {well}: missing typewell, skip"); continue
        h = pd.read_csv(h_path)
        t = pd.read_csv(t_path)
        ps = detect_ps(h)
        true_tvt = h["TVT"].to_numpy()
        rec = {"well": well, "n_h": len(h), "n_t": len(t), "ps": ps, "n_pred": len(h) - ps}
        for name, fn in PREDICTORS.items():
            pred = fn(h, t, ps)
            rec[f"rmse_{name}"] = post_ps_rmse(true_tvt, pred, ps)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if len(df):
        print(f"\nMean RMSE over {len(df)} wells:")
        for k in PREDICTORS:
            print(f"  {k:6s}: {df[f'rmse_{k}'].mean():.4f}")
