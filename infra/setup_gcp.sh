#!/usr/bin/env bash
# One-time GCP setup. Run from local machine with gcloud already authenticated.
# Edit the values below before running.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-CHANGE_ME}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET="${GCS_BUCKET:-progol-data-storage}"
SA_NAME="progol-runner"
VM_NAME="progol-trainer"
REPO_URL="${REPO_URL:-https://github.com/EIvan21/progol_predictions.git}"
FOOTBALL_API_KEY="${FOOTBALL_API_KEY:-CHANGE_ME}"

if [[ "$PROJECT_ID" == "CHANGE_ME" ]]; then
  echo "Set GCP_PROJECT_ID env var first." >&2
  exit 1
fi

gcloud config set project "$PROJECT_ID"

echo "==> Enabling required APIs..."
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  cloudscheduler.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com

echo "==> Creating GCS bucket: gs://$BUCKET ..."
gsutil mb -l "$REGION" -c STANDARD "gs://$BUCKET" || echo "(bucket may already exist)"
gsutil lifecycle set "$(dirname "$0")/lifecycle.json" "gs://$BUCKET"

echo "==> Storing API key + startup script in GCS..."
echo "FOOTBALL_API_KEY=$FOOTBALL_API_KEY" > /tmp/.env.progol
echo "USE_GCS=true" >> /tmp/.env.progol
echo "GCS_BUCKET=$BUCKET" >> /tmp/.env.progol
echo "LOG_JSON=true" >> /tmp/.env.progol
gsutil cp /tmp/.env.progol "gs://$BUCKET/secrets/.env"
gsutil cp "$(dirname "$0")/startup.sh" "gs://$BUCKET/scripts/startup.sh"
rm -f /tmp/.env.progol

echo "==> Creating service account: $SA_NAME ..."
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Progol training runner" || true

SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role=roles/storage.objectAdmin
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role=roles/compute.instanceAdmin.v1

echo "==> Creating Spot VM: $VM_NAME ..."
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type=e2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-standard \
  --service-account="$SA_EMAIL" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata="enable-oslogin=TRUE,startup-script-url=gs://$BUCKET/scripts/startup.sh,gcs-bucket=$BUCKET,repo-url=$REPO_URL" \
  --metadata-from-file=startup-script="$(dirname "$0")/startup.sh" || \
  echo "(VM may already exist — to recreate: gcloud compute instances delete $VM_NAME --zone=$ZONE)"

# Stop immediately after creation; Scheduler will start it on a cron.
gcloud compute instances stop "$VM_NAME" --zone="$ZONE" || true

echo "==> Done. Next steps:"
echo "   1) Schedule weekly run:    bash infra/scheduler.sh"
echo "   2) Trigger a one-off run:  gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "   3) View logs:              gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE"
echo "   4) Tear down:              bash infra/teardown.sh"
