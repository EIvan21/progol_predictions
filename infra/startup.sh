#!/usr/bin/env bash
# Runs at VM boot. Pulls repo, syncs DB from GCS, runs the pipeline, syncs
# results back, then shuts down.
set -euo pipefail
exec > /var/log/progol-startup.log 2>&1

BUCKET="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
upload_log() {
  gsutil cp /var/log/progol-startup.log "gs://$BUCKET/logs/startup-$RUN_ID.log" || true
}
trap upload_log EXIT

REPO_URL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url)"
PROGOL_BUDGET="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/progol-budget || echo "")"
WORK_DIR="/opt/progol_predictions"

apt-get update -y
apt-get install -y python3 python3-venv python3-pip git curl

if [ ! -d "$WORK_DIR/.git" ]; then
  git clone "$REPO_URL" "$WORK_DIR"
else
  git -C "$WORK_DIR" pull --ff-only
fi

cd "$WORK_DIR"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data data/processed models reports logs

gsutil cp "gs://$BUCKET/secrets/.env" .env || echo "no secrets/.env in bucket"
gsutil -m rsync -r "gs://$BUCKET/db" data/ || echo "no db prefix in bucket"
gsutil -m rsync -r "gs://$BUCKET/models" models/ || echo "no models prefix in bucket"

export USE_GCS=true
export GCS_BUCKET="$BUCKET"
export LOG_JSON=true
[ -n "$PROGOL_BUDGET" ] && export PROGOL_BUDGET="$PROGOL_BUDGET"

python -m src.progol.ingest.fetch_data
python -m src.progol.modeling.preprocess
python -m src.progol.modeling.tune --trials 30 --timeout 1200 || echo "tune skipped"
python -m src.progol.modeling.train
python -m src.progol.modeling.walk_forward --folds 6 || true
python -m src.progol.modeling.backtest --kelly 0.25 --min-edge 0.04 || true

python -m src.progol.reporting.eda || echo "eda generation failed"
python -m src.progol.reporting.generate_report || echo "training summary failed"
python -m src.progol.ingest.get_progol_ids || echo "progol slate scrape failed"
python -m src.progol.modeling.predict || echo "prediction step failed"

gsutil -m rsync -r data "gs://$BUCKET/db"
gsutil -m rsync -r models "gs://$BUCKET/models"
gsutil -m rsync -r reports "gs://$BUCKET/reports" || true
gsutil cp current_progol_ids.json "gs://$BUCKET/predictions/slate-$RUN_ID.json" || true

shutdown -h now
