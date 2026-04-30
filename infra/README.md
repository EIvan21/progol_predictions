# GCP infra for progol_predictions

Spot-VM weekly trainer. Persists data + models in a single GCS bucket, runs the
full pipeline on schedule, then auto-shuts down. **~USD $5–6/month** for
compute + storage (api-football plan billed separately).

## Architecture

```
GCS bucket (progol-data-storage)
├── secrets/.env              <- API keys
├── db/progol.db              <- match history
├── db/predictions.db         <- prediction log
├── models/                   <- versioned models + latest.json
├── reports/                  <- EDA PDFs
└── logs/                     <- VM startup logs

Cloud Scheduler (cron: 0 6 * * MON)
        │
        ▼ start instance
Spot VM (e2-standard-4) ── runs startup.sh ── shuts itself off
```

## Prerequisites

- Local: `gcloud` authenticated (`gcloud auth login` + `gcloud auth application-default login`)
- An api-football.com plan (Pro $19/mo recommended)
- A GCP project with billing enabled

## Setup (one time)

```bash
export GCP_PROJECT_ID=your-project-id
export FOOTBALL_API_KEY=your-api-sports-key
export GCS_BUCKET=progol-data-storage         # optional, has a default
export REPO_URL=https://github.com/EIvan21/progol_predictions.git

bash infra/setup_gcp.sh
bash infra/scheduler.sh        # weekly run, Mondays 06:00 MX time
```

## Manual one-off run

```bash
gcloud compute instances start progol-trainer --zone=us-central1-a
# the VM runs startup.sh and stops itself when done (~30–60 min)
```

## Inspect

```bash
gcloud compute instances get-serial-port-output progol-trainer --zone=us-central1-a
gsutil ls gs://progol-data-storage/models/
gsutil cat gs://progol-data-storage/models/latest.json
```

## Cost levers

| Lever | Effect |
|---|---|
| **Spot VM** (default) | -70% on compute |
| **`shutdown -h now` in startup** (default) | pay only while training |
| `pd-standard` instead of `pd-balanced` | -25% on disk |
| Lifecycle rule: NEARLINE > 30d, COLDLINE > 90d | cheaper old artifacts |
| Cloud Scheduler 3 jobs free | scheduler is free |

## Tear down

```bash
bash infra/teardown.sh                  # keeps bucket
bash infra/teardown.sh --delete-bucket  # nukes everything
```

## What runs in startup.sh

1. Sync `db/` and `models/` from GCS
2. `python -m src.progol.ingest.fetch_data`
3. `python -m src.progol.modeling.preprocess`
4. `python -m src.progol.modeling.tune` (Optuna, 30 trials max)
5. `python -m src.progol.modeling.train`
6. `python -m src.progol.modeling.walk_forward`
7. `python -m src.progol.modeling.backtest`
8. Sync `data/`, `models/`, `reports/`, `logs/` back to GCS
9. `shutdown -h now`
