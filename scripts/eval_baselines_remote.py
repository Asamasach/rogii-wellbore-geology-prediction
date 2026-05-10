"""Run baselines on all 773 train wells on fteam6, also sweep align params."""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import time
import numpy as np
import pandas as pd

# In-place import of the predictors module living next to this script on fteam6
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_tvt import (
    predict_const, predict_linear, predict_align, post_ps_rmse, detect_ps,
)

ROOT = Path("/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw")
TRAIN_DIR = ROOT / "train"

print(f"=== full evaluation on all train wells ===", flush=True)
horizontal_files = sorted(TRAIN_DIR.glob("*horizontal_well.csv"))
print(f"found {len(horizontal_files)} horizontal_well files", flush=True)

# 1) Run const + linear over all wells (fast, ~ms per well)
rows = []
t0 = time.time()
for i, hp in enumerate(horizontal_files):
    well = hp.name.split("__")[0]
    tp = TRAIN_DIR / f"{well}__typewell.csv"
    if not tp.exists():
        continue
    h = pd.read_csv(hp)
    t = pd.read_csv(tp)
    ps = detect_ps(h)
    true_tvt = h["TVT"].to_numpy()
    rec = {"well": well, "n_h": len(h), "n_t": len(t), "ps": ps, "n_pred": len(h) - ps}
    rec["rmse_const"] = post_ps_rmse(true_tvt, predict_const(h, t, ps), ps)
    rec["rmse_linear"] = post_ps_rmse(true_tvt, predict_linear(h, t, ps), ps)
    rows.append(rec)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(horizontal_files)}  t={time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
print(f"\n=== const + linear over {len(df)} wells ===")
print(f"mean RMSE const : {df['rmse_const'].mean():.3f}  median {df['rmse_const'].median():.3f}  "
      f"p90 {df['rmse_const'].quantile(0.9):.3f}")
print(f"mean RMSE linear: {df['rmse_linear'].mean():.3f}  median {df['rmse_linear'].median():.3f}  "
      f"p90 {df['rmse_linear'].quantile(0.9):.3f}")

# 2) Sweep align parameters on a 60-well subset (it's slow, O(n_h * gr_win) per well)
print(f"\n=== align param sweep on 60-well random subset ===", flush=True)
np.random.seed(0)
subset = list(np.random.choice(horizontal_files, size=60, replace=False))
configs = [
    dict(gr_win=25, max_step=0.5),
    dict(gr_win=25, max_step=1.0),
    dict(gr_win=25, max_step=2.0),
    dict(gr_win=51, max_step=1.0),
    dict(gr_win=51, max_step=2.0),
    dict(gr_win=101, max_step=1.0),
    dict(gr_win=101, max_step=2.0),
]
for cfg in configs:
    t0 = time.time()
    rmses = []
    for hp in subset:
        well = hp.name.split("__")[0]
        tp = TRAIN_DIR / f"{well}__typewell.csv"
        if not tp.exists():
            continue
        h = pd.read_csv(hp)
        t = pd.read_csv(tp)
        ps = detect_ps(h)
        true_tvt = h["TVT"].to_numpy()
        pred = predict_align(h, t, ps, **cfg)
        rmses.append(post_ps_rmse(true_tvt, pred, ps))
    rmses = np.array(rmses)
    print(f"  cfg={cfg} mean={rmses.mean():.3f} median={np.median(rmses):.3f} "
          f"p90={np.quantile(rmses, 0.9):.3f} t={time.time()-t0:.0f}s",
          flush=True)
