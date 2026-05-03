#!/usr/bin/env bash
# Runs at VM boot. Pulls repo, syncs DB from GCS, runs the pipeline, syncs
# results back, then shuts down.
set -euo pipefail
exec > /var/log/progol-startup.log 2>&1

BUCKET="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="/opt/progol_predictions"
finalize() {
  # Run on every script exit (success, set -e abort, signal). Sync any
  # produced artifacts to GCS, upload the log, then halt. Any individual
  # failure here must not block subsequent uploads — disable -e and guard
  # each call. Halt only after all uploads complete to avoid the kernel
  # racing the network teardown.
  set +e
  cd "$WORK_DIR" 2>/dev/null
  gsutil -m rsync -r data "gs://$BUCKET/db"                     || true
  gsutil -m rsync -r models "gs://$BUCKET/models"                || true
  gsutil -m rsync -r reports "gs://$BUCKET/reports"              || true
  # Python's gcs.upload_file fails on GCE Metadata Server mTLS in this image
  # (see train.py / predict.py warnings), so latest.json + slate_*.json don't
  # land via that path. Sync the whole predictions/ dir via gcloud auth.
  if [ -d predictions ]; then
    gsutil -m rsync -r predictions "gs://$BUCKET/predictions"   || true
  fi
  if [ -f current_progol_ids.json ]; then
    gsutil cp current_progol_ids.json "gs://$BUCKET/predictions/slate-$RUN_ID.json" || true
  fi
  gsutil cp /var/log/progol-startup.log "gs://$BUCKET/logs/startup-$RUN_ID.log" || true
  sync
  shutdown -h now
}
trap finalize EXIT

REPO_URL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url)"
PROGOL_BUDGET="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/progol-budget || echo "")"

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

# GCE Metadata Server now uses mTLS on HTTPS:443. The system CA bundle has
# the freshly-bootstrapped MDS root cert, but Python's requests/certifi does
# not. Point Python at the system bundle so google-auth can talk to MDS.
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

python -m src.progol.ingest.fetch_data
python -m src.progol.modeling.preprocess
python -m src.progol.modeling.tune --trials 30 --timeout 1200 || echo "tune skipped"
python -m src.progol.modeling.train
python -m src.progol.modeling.walk_forward --folds 6 || true
python -m src.progol.modeling.backtest --kelly 0.25 --min-edge 0.04 || true

python -m src.progol.reporting.eda || echo "eda generation failed"
python -m src.progol.reporting.generate_report || echo "training summary failed"
# Slate scrape MUST succeed to predict the right matches. Don't mask failure.
python -m src.progol.ingest.get_progol_ids
python -m src.progol.modeling.predict || echo "prediction step failed"

# Uploads + shutdown happen in the EXIT trap (finalize) defined above.
