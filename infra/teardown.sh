#!/usr/bin/env bash
# Removes the VM, scheduler job, and (optionally) bucket. Asks for confirmation.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?must set GCP_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
REGION="${GCP_REGION:-us-central1}"
VM_NAME="${VM_NAME:-progol-trainer}"
JOB_NAME="${JOB_NAME:-progol-weekly}"
BUCKET="${GCS_BUCKET:-progol-data-storage}"

echo "About to delete:"
echo "  - VM: $VM_NAME ($ZONE)"
echo "  - Scheduler job: $JOB_NAME ($REGION)"
echo "  - Bucket gs://$BUCKET will be PRESERVED. Add --delete-bucket to remove it."
read -p "Continue? (y/N) " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0

gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet || true
gcloud scheduler jobs delete "$JOB_NAME" --location="$REGION" --quiet || true

if [[ "${1:-}" == "--delete-bucket" ]]; then
  gsutil -m rm -r "gs://$BUCKET" || true
fi
