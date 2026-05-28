---
name: Bot reads stale data unless command syncs
description: Only /predecir_progol calls _gcs_sync(). Other commands hit local DB/predictions which can lag the trainer's last upload by days.
type: project
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
The bot's `app.py` only invokes `_gcs_sync()` from `cmd_predecir_progol` and at process startup. `/presupuesto`, `/predecir_partido`, `/ultima_prediccion_progol` all read whatever is currently on disk in `/opt/progol_predictions/{data,predictions}`.

**Why:** sync is a 120s-timeout subprocess call to `gsutil rsync` — making every command sync is expensive. The current intent is that /predecir_progol is the "fresh" command; others trust the most recent sync.

**How to apply:**
- After a manual trainer run or first-time fix, manually sync the bot before testing:
  `gcloud compute ssh progol-bot --zone=us-central1-a --project=progol-predictor --command="sudo -u progol-bot gsutil -m rsync -r gs://progol-data-storage/predictions /opt/progol_predictions/predictions/ && sudo -u progol-bot gsutil -m rsync -r gs://progol-data-storage/db /opt/progol_predictions/data/"`
- The trainer's `predict.py` only uploads `predictions.db` to `predictions/`. The main `progol.db` (which carries `progol_concurso_games.predicted_probs`) is uploaded by startup.sh's `finalize` trap (`gsutil rsync -r data gs://$BUCKET/db`). If you bypass startup.sh, run that rsync manually or the bot will keep showing stale per-match rows.
- The bot's slim venv was missing numpy before 2026-05-15; `quiniela.optimize_budget` now imports it at module load (via `app.py`'s `from src.progol.modeling.quiniela import BASE_COST_MXN`). Already in `requirements_bot.txt`.
