# ROGII — Wellbore Geology Prediction

Work-in-progress on the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) Kaggle competition.

**Task:** for each horizontal-well horizontal step after a Prediction-Start point, predict TVT (true vertical thickness) using:
- the horizontal well's `MD/X/Y/Z/GR` sequence + `TVT_input` (TVT known up to PS)
- a matched typewell's full `TVT/GR/Geology` log
- 6 formation-top depths in train horizontals (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA)

Metric: RMSE on `(predicted − actual)` over all post-PS rows of the 3 test wells.

## Repository contents

```
.
├── scripts/                 # local + remote helpers
│   ├── predict_tvt.py            # const + linear baselines
│   ├── predict_tvt_pf.py         # particle filter (research; not used standalone)
│   ├── predict_tvt_lgb.py        # v1 LGB residual features (29)
│   ├── predict_tvt_lgb_v2.py     # v2 + NCC alignment features (44)
│   ├── predict_tvt_lgb_v3.py     # v3 + formation plane-fit features (62)
│   ├── eval_*_remote.py          # GroupKFold(5) eval scripts run on remote box
│   ├── eval_catboost_remote.py   # CatBoost OOF + Ridge stack
│   ├── postproc_search_v3_remote.py  # alpha/tau/SavGol grid search
│   ├── ship_*.py / migrate_*.py  # SSH-based deployment helpers (env-var creds)
│   └── ...
├── kernel/, kernel_const/, kernel_lgb/, kernel_lgb_v3/, kernel_lgb_v4/  # Kaggle kernel packages (mirror, const, LGB, LGB+CAT)
└── kernel_v9956/                 # fork of romantamrazov public sub-9.956 notebook
                                  # (notebook itself excluded; see kernel-metadata.json for attribution)
```

## Reproducing

The competition data is **not** redistributed in this repo. To reproduce:

1. Accept the comp rules at https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
2. Download the data (Kaggle CLI):
   ```
   kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw/
   ```
3. The remote-execution helpers (`scripts/migrate_to_fteam11.py`, etc.) read connection info from env vars — set:
   ```
   export FTEAM6_HOST=...   FTEAM6_PORT=...   FTEAM6_USER=...   FTEAM6_PASS=...
   export FTEAM11_HOST=...  FTEAM11_PORT=...  FTEAM11_USER=...  FTEAM11_PASS=...
   export KAGGLE_API_TOKEN=KGAT_...
   ```

## Approach

Best original score so far: **LB 14.454** (LGB v1 with 29 simple features on residual `TVT − last_known_tvt`, GroupKFold(5) by well).

The `kernel_v9956/` directory is a fork of [romantamrazov's "BETTER SOLUTION | LB: 9.956"](https://www.kaggle.com/code/romantamrazov/rogii-better-solution-lb-9-956) public Kaggle notebook used as a strong baseline to iterate on top of.

## Acknowledgements

- Public ROGII notebooks by Roman Tamrazov, shinyanagai123, nina2025, and others on Kaggle for the algorithmic patterns (particle-filter alignment, formation plane-fit KNN, GBDT residual stacking).
