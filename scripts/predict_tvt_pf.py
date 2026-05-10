"""TVT particle filter — core building block from public SOTA notebooks.

State per particle: (TVT, dTVT). Transition: dTVT' = dTVT + N(0, sigma_a),
TVT' = TVT + dTVT (per 1-foot MD step). Observation: at horizontal step i,
likelihood = N(hw_gr[i] | tw_gr_at(particle.TVT), sigma_gr).

Calibrated from the prefix (TVT_input known zone): sigma_gr from residuals
between hw_gr[i] and tw_gr_at(TVT_input[i]); sigma_a from observed dTVT
spread in calibration.

Initialised at PS with particles spread around TVT_input[PS-1] using prefix
dTVT statistics. Systematic resampling when Neff < 0.5N + small roughening.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

from predict_tvt import detect_ps, fillna_gr, post_ps_rmse


def _interp(t_tvt: np.ndarray, t_gr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised linear interpolation of t_gr (sampled at t_tvt) at points x.

    t_tvt must be monotonically non-decreasing. x is broadcast-friendly.
    Out-of-range x clamps to endpoints.
    """
    return np.interp(x, t_tvt, t_gr)


def predict_pf(
    h: pd.DataFrame, t: pd.DataFrame, ps: int,
    n_particles: int = 300,
    sigma_a_default: float = 0.05,        # accel (dTVT change) std per step
    sigma_gr_default: float = 25.0,       # GR likelihood std (clipped after calib)
    rough_sigma_tvt: float = 0.25,
    rough_sigma_d: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Return per-row predicted TVT (calibration zone left as TVT_input)."""
    out = h["TVT_input"].to_numpy().copy().astype(np.float64)
    n_h = len(h)
    if ps == 0 or ps >= n_h:
        return out

    h_gr = fillna_gr(h["GR"].to_numpy())
    t_tvt = t["TVT"].to_numpy()
    t_gr = fillna_gr(t["GR"].to_numpy())
    if not np.all(np.diff(t_tvt) >= 0):
        # Sort typewell by TVT just in case
        order = np.argsort(t_tvt)
        t_tvt = t_tvt[order]; t_gr = t_gr[order]

    rng = np.random.default_rng(seed)

    # --- Calibrate from prefix ---
    # GR sigma: residuals between hw_gr[i] and tw_gr_at(TVT_input[i])
    pre = slice(0, ps)
    pre_tvt = h["TVT_input"].iloc[pre].to_numpy()
    pre_hg = h_gr[pre]
    expected_gr = _interp(t_tvt, t_gr, pre_tvt)
    valid = np.isfinite(pre_hg) & np.isfinite(expected_gr)
    if valid.sum() >= 20:
        sigma_gr = float(np.clip(np.std(pre_hg[valid] - expected_gr[valid]), 8.0, 60.0))
    else:
        sigma_gr = sigma_gr_default

    # dTVT spread per step
    if ps >= 5:
        dtvt = np.diff(pre_tvt)
        sigma_d = float(np.clip(np.std(dtvt), 0.001, 0.2))
        d_mean = float(np.mean(dtvt[-min(50, len(dtvt)):]))  # local trend at end of prefix
    else:
        sigma_d = sigma_a_default
        d_mean = 0.0

    sigma_a = max(sigma_d * 0.3, 0.01)  # allow dTVT to walk

    # --- Init particles around TVT_input[PS-1], dTVT around d_mean ---
    init_tvt = float(pre_tvt[-1])
    tvt = rng.normal(init_tvt, max(sigma_d, 0.2), size=n_particles)
    dtvt = rng.normal(d_mean, max(sigma_d, 0.005), size=n_particles)
    log_w = np.zeros(n_particles)

    INV2 = 1.0 / (2 * sigma_gr * sigma_gr)
    LOG_NORM = -0.5 * np.log(2 * np.pi * sigma_gr * sigma_gr)
    half_N = n_particles * 0.5

    for i in range(ps, n_h):
        # transition
        dtvt = dtvt + rng.normal(0.0, sigma_a, size=n_particles)
        tvt = tvt + dtvt

        # observation
        gr_i = h_gr[i]
        if np.isfinite(gr_i):
            expected = np.interp(tvt, t_tvt, t_gr)
            log_lik = LOG_NORM - (gr_i - expected) ** 2 * INV2
            log_w = log_w + log_lik
        # normalise
        c = log_w.max()
        w = np.exp(log_w - c)
        wsum = w.sum()
        if wsum > 0:
            w /= wsum
        else:
            w = np.ones(n_particles) / n_particles

        # estimate
        out[i] = float(np.sum(w * tvt))

        # resample if Neff < half N
        neff = 1.0 / np.sum(w * w)
        if neff < half_N:
            # systematic resampling
            positions = (np.arange(n_particles) + rng.random()) / n_particles
            cum = np.cumsum(w)
            cum[-1] = 1.0
            idx = np.searchsorted(cum, positions)
            tvt = tvt[idx]; dtvt = dtvt[idx]
            # roughening
            tvt = tvt + rng.normal(0.0, rough_sigma_tvt, size=n_particles)
            dtvt = dtvt + rng.normal(0.0, rough_sigma_d, size=n_particles)
            log_w = np.zeros(n_particles)
        else:
            # store back as log-weights for next iter (avoid recomputing)
            log_w = np.log(np.clip(w, 1e-300, None))

    return out


def pooled_rmse(true_tvt_list, pred_tvt_list, ps_list):
    """Pooled RMSE across all eval rows (matches Kaggle scoring)."""
    errs = []
    for tt, pp, ps in zip(true_tvt_list, pred_tvt_list, ps_list):
        if ps < len(tt):
            errs.append(tt[ps:] - pp[ps:])
    if not errs:
        return float("nan")
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


# ------------------- driver: local 4-sample test -----------------------
if __name__ == "__main__":
    SAMPLES = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\_samples_train")
    rows = []
    true_list, pred_list, ps_list = [], [], []
    pred_const_list = []
    for hp in sorted(SAMPLES.glob("*horizontal_well.csv")):
        well = hp.name.split("__")[0]
        tp = SAMPLES / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp); t = pd.read_csv(tp)
        ps = detect_ps(h)
        true = h["TVT"].to_numpy()

        # const baseline reference
        from predict_tvt import predict_const
        pc = predict_const(h, t, ps)
        # PF prediction
        pp = predict_pf(h, t, ps, n_particles=300, seed=0)

        rec = {
            "well": well, "n_h": len(h), "n_t": len(t), "ps": ps, "n_pred": len(h) - ps,
            "rmse_const": post_ps_rmse(true, pc, ps),
            "rmse_pf": post_ps_rmse(true, pp, ps),
        }
        rows.append(rec)
        true_list.append(true); pred_list.append(pp); ps_list.append(ps); pred_const_list.append(pc)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nMean per-well RMSE   const: {df['rmse_const'].mean():.3f}  pf: {df['rmse_pf'].mean():.3f}")
    print(f"Median per-well RMSE const: {df['rmse_const'].median():.3f}  pf: {df['rmse_pf'].median():.3f}")
    print(f"Pooled RMSE          const: {pooled_rmse(true_list, pred_const_list, ps_list):.3f}  "
          f"pf: {pooled_rmse(true_list, pred_list, ps_list):.3f}")
