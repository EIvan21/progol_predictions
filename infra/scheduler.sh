#!/usr/bin/env bash
# Schedules a weekly auto-start of the trainer VM via Cloud Scheduler.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?must set GCP_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
REGION="${GCP_REGION:-us-central1}"
VM_NAME="${VM_NAME:-progol-trainer}"
SA_EMAIL="progol-runner@$PROJECT_ID.iam.gserviceaccount.com"
JOB_NAME="${JOB_NAME:-progol-weekly}"
CRON="${CRON:-0 6 * * MON}"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

# Allow Scheduler to act as the SA when calling Compute API
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-cloudscheduler.iam.gserviceaccount.com" \
  --role=roles/iam.serviceAccountTokenCreator || true

gcloud scheduler jobs create http "$JOB_NAME" \
  --location="$REGION" \
  --schedule="$CRON" \
  --time-zone="America/Mexico_City" \
  --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$VM_NAME/start" \
  --http-method=POST \
  --oauth-service-account-email="$SA_EMAIL" || \
  gcloud scheduler jobs update http "$JOB_NAME" \
    --location="$REGION" \
    --schedule="$CRON" \
    --time-zone="America/Mexico_City" \
    --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$VM_NAME/start" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL"

echo "Scheduled '$JOB_NAME' at '$CRON' (America/Mexico_City)."
