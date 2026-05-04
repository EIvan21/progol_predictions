#!/usr/bin/env bash
# Boot script for the progol-bot VM. Idempotent so reboots reapply.
# Pulls repo, installs deps, fetches secrets/state from GCS, installs the
# systemd unit, starts the long-polling bot.
set -euo pipefail
exec > /var/log/progol-bot-startup.log 2>&1

BUCKET="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)"
REPO_URL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url)"

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
# Bot only needs a slim subset (no pandas/sklearn/xgboost). Saves ~1.3GB
# of disk + many minutes of install on e2-micro.
pip install -r requirements_bot.txt

gsutil cp "gs://$BUCKET/secrets/.env" "$WORK_DIR/.env" || echo "no .env in bucket"

mkdir -p data predictions
gsutil -m rsync -r "gs://$BUCKET/db" data/ || true
gsutil -m rsync -r "gs://$BUCKET/predictions" predictions/ || true

id -u progol-bot &>/dev/null || useradd --system --home-dir "$WORK_DIR" --shell /usr/sbin/nologin progol-bot
chown -R progol-bot:progol-bot "$WORK_DIR"

cp "$WORK_DIR/infra/progol-bot.service" /etc/systemd/system/progol-bot.service
systemctl daemon-reload
systemctl enable progol-bot.service
systemctl restart progol-bot.service
