#!/usr/bin/env bash
# One-shot provision for the progol-bot e2-micro VM. Free-tier eligible
# in us-central1, us-west1, us-east1. Reuses progol-runner SA so the bot
# can rsync the same bucket the trainer writes.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-progol-predictor}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET="${GCS_BUCKET:-progol-data-storage}"
SA_EMAIL="progol-runner@$PROJECT_ID.iam.gserviceaccount.com"
VM_NAME="progol-bot"
REPO_URL="${REPO_URL:-https://github.com/EIvan21/progol_predictions.git}"

gcloud config set project "$PROJECT_ID"

echo "==> Creating $VM_NAME (e2-micro) in $ZONE ..."
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=10GB \
  --boot-disk-type=pd-standard \
  --service-account="$SA_EMAIL" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata="gcs-bucket=$BUCKET,repo-url=$REPO_URL" \
  --metadata-from-file=startup-script="$(dirname "$0")/bot_startup.sh"

echo "==> Done. Watch boot:"
echo "   gcloud compute instances tail-serial-port-output $VM_NAME --zone=$ZONE"
echo "==> Service status (after ~3 min):"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE -- sudo systemctl status progol-bot"
