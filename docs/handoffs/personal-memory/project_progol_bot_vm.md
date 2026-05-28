---
name: progol-bot VM provisioning
description: Separate e2-micro VM for the long-polling Telegram bot, slim deps, DB-backed auth via bot_users.
type: project
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
The interactive Telegram bot lives on its own GCE VM (`progol-bot`, e2-micro, us-central1-a, free tier). Independent from the weekly trainer (`progol-trainer`).

**Why:** the trainer VM only runs Wednesdays and shuts down. The bot needs to answer commands 24/7. Splitting them lets each be sized correctly and isolates blast radius.

**How to apply:**
- VM uses `requirements_bot.txt`, NOT the full `requirements.txt`. The full ML stack (pandas/sklearn/xgboost/catboost/lightgbm/scipy + nvidia-nccl-cu12 transitive) is ~1.5GB and OOMs / takes 10+ min on e2-micro. The slim file is ~30MB. `database.py` lazy-imports pandas inside `get_all_matches_df()` so module load doesn't require it.
- Bot is `systemd`-managed via `infra/progol-bot.service` (User=`progol-bot`, `WorkingDirectory=/opt/progol_predictions`, `EnvironmentFile=.env`). Restart on failure.
- Auth is **DB-backed** in `bot_users` (chat_id PK, user_id, role, status). Lives in **`data/bot.db`** (NOT `data/progol.db`) because the bot's `gsutil rsync` from GCS overwrites `progol.db` with the trainer's version on every restart, which would silently wipe `bot_users`. `bot.db` is local-only and never synced. Owner seeded once via SSH:
  `cd /opt/progol_predictions && sudo -u progol-bot ./venv/bin/python -m src.progol.bot.manage_users add CHAT_ID --role owner`
- `/start` and `/whoami` are public; everything else passes `_guard('user')`. Admin commands `/usuarios /aprobar /bloquear` need owner|admin. New users register as `pending` automatically.
- `bot_threads` table exists for Phase 2 conversation state (budget flow). Not used in Phase 1.
- VM reads predictions/db from `gs://progol-data-storage` via `gsutil rsync` (one-way). `google-auth` Python SDK fails on GCE mTLS metadata server, same as trainer; `gsutil` works.
- When invoking `manage_users` CLI manually, **must** `cd /opt/progol_predictions` first or `python -m src.progol.bot.*` fails with ModuleNotFoundError.
