---
name: progol-trainer VM provisioning
description: progol-trainer GCE VM was originally Spot/preemptible; that silently truncated full pipeline runs. Now configured as STANDARD.
type: project
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
The `progol-trainer` GCE VM (us-central1-a, e2-standard-4, project `progol-predictor`) was originally created with `provisioningModel=SPOT, preemptible=True`. Pipeline failure mode: GCE preempted the VM mid-run with ~30s of grace, which was not enough time to complete the final `gsutil rsync` of the 100 MB `calibrated_ensemble.pkl`. Result: VM terminated cleanly but bucket received no artifacts and no log. Caused multiple confused debug cycles where each run looked like a different bug (race condition, SSL, etc) when the real cause was preemption.

**Why:** The pipeline writes a 100 MB stacking model + ~1 MB EDA PDF + per-fold artifacts at the END of the run. Spot preemption SIGTERM → 30s grace → SIGKILL is not enough to flush all uploads.

**How to apply:** Before running an end-to-end pipeline on `progol-trainer`, confirm it's STANDARD: `gcloud compute instances describe progol-trainer --zone=us-central1-a --format="value(scheduling.preemptible,scheduling.provisioningModel)"` should print `False    STANDARD`. If switching back to Spot for cost savings, redesign the pipeline to upload artifacts incrementally (after each major step) rather than once at end. Toggle command: `gcloud compute instances set-scheduling progol-trainer --zone=us-central1-a --provisioning-model=STANDARD --no-preemptible --clear-instance-termination-action --maintenance-policy=MIGRATE` (must be stopped first; `--maintenance-policy=MIGRATE` is required because e2 VMs reject `onHostMaintenance=TERMINATE` on standard).
