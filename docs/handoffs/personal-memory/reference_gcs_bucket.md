---
name: progol-data-storage GCS bucket layout
description: Layout and contract of gs://progol-data-storage — input data, model artifacts, reports, predictions, logs.
type: reference
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
`gs://progol-data-storage` is the single bucket the pipeline reads from and writes to. Layout:

- `db/` — SQLite (`progol.db`) + processed CSVs. Pulled into VM `data/` at boot, rsynced back at shutdown.
- `db/raw/progol_history.csv` — long-lived seed file (Quiniela history).
- `models/` — final stacking model (`calibrated_ensemble.pkl`, ~100 MB), `metrics.json`, `latest.json`, `feature_stats.json`, `best_params.json`, `walk_forward.json`, `backtest.json`. Versioned subdirs `v_YYYYMMDD_HHMMSS/` keep prior trainings.
- `reports/` — `eda_*.pdf`, `experiment_PROD_CONTEXTUAL_*.tex`, `mutual_information.csv`, `vif_analysis.csv`, plus `eda_plots/*.png`.
- `predictions/slate-<RUN_ID>.json` — per-run Progol slate match IDs from `get_progol_ids` + model probabilities from `predict.py`.
- `logs/startup-<RUN_ID>.log` — per-run pipeline log uploaded by `infra/startup.sh`'s `finalize` EXIT trap.
- `scripts/startup.sh` — VM startup-script-url target. The VM also has the script set inline as `startup-script` metadata; keep the two in sync.
- `secrets/.env` — copied into the VM workdir at boot.

Auth note: the bucket is accessed two ways from the VM. (1) `gsutil` uses `gcloud` auth and works. (2) Python `google-cloud-storage` uses `google-auth` which hits the GCE Metadata Server's mTLS HTTPS endpoint and fails certifi verification on the current Debian 12 image even with `REQUESTS_CA_BUNDLE` set. The pipeline routes around this by making `train.py`'s python upload non-fatal and relying on the bash `gsutil rsync` block in startup.sh's `finalize` trap.
