# CLAUDE.md — Progol Predictions

Reference for Claude Code sessions working on this repo.

## Project overview

ML pipeline that predicts results for Mexico's **Progol** sports-betting game (21 fixtures per concurso: 14 main + 7 revancha). Output: per-fixture L/E/V probabilities → top-10 quinielas via Monte Carlo + a budget-optimized plan → posted to a Telegram bot.

## Architecture at a glance

```
fetch_data.py  → preprocess.py  → tune.py → train.py → walk_forward → backtest
                                                          ↓
get_progol_ids.py  → predict.py  → predictions/latest.json + concurso DB rows
                                          ↓
                              bot/send_predictions.py → Telegram
                              bot/app.py (long-polling, /historial etc.)
```

The whole pipeline runs on a **GCE VM (`progol-trainer`)** at boot via `infra/startup.sh`: it pulls the repo, runs every step, syncs artifacts to GCS, and shuts itself down. The **bot VM (`progol-bot`)** runs systemd-managed long-polling, syncing from GCS on demand.

## Key directories

| Path | Purpose |
|------|---------|
| `src/progol/ingest/` | API-Football fetcher, Progol scraper, slate resolver |
| `src/progol/modeling/` | preprocess, tune (Optuna), train (stacked ensemble + Dirichlet calibration), predict, walk_forward, backtest, quiniela optimizer |
| `src/progol/features/` | `team_state.py` — inference-time feature builder (must mirror training transforms) |
| `src/progol/bot/` | Telegram bot: `app.py` long-polling, `send_predictions.py` weekly push, `formatting.py` HTML rendering |
| `src/progol/reporting/` | EDA, league dashboard, **progol_history** (per-concurso hit-rate) |
| `src/progol/storage/` | versioning, GCS upload helpers |
| `src/progol/database.py` | SQLite schema + helpers (concurso games, predictions, settle_actuals) |
| `src/progol/config.py` | Paths, league IDs, per-league market blend, cup IDs |
| `infra/startup.sh` | GCE VM bootstrap — full pipeline + Telegram + shutdown |
| `tests/` | pytest suite, run with plain `pytest` |
| `predictions/` | `latest.json` (current concurso) + dated `slate_*.json` |
| `data/` | SQLite DBs (`progol.db`, `predictions.db`, `bot.db`) — **gitignored** |
| `models/` | Versioned model bundles `v_YYYYMMDD_HHMMSS/calibrated_ensemble.pkl` — **gitignored** |

## Environment / setup

- **Python**: pinned to **3.11** in `.python-version` (matches trainer VM Debian bookworm `python3 3.11.2`). Local dev on 3.10.x works for most tests but production targets 3.11.
- **Virtualenv**: `python -m venv venv && source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows PowerShell). `setup.sh` / `setup.ps1` exist as helpers.
- **Pip install**: `pip install -r requirements.txt` for the full ML stack. The bot VM uses `requirements_bot.txt` (slim — no sklearn/xgboost/etc., but still needs numpy for `quiniela.optimize_budget`).
- **Windows SSL workaround for pip**: `pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <pkg>`.
- **Secrets**: `.env` is gitignored. Required keys (see `.env.example`): `FOOTBALL_API_KEY` (API-Football), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. On the trainer VM, the `.env` is fetched from `gs://progol-data-storage/secrets/.env` during `infra/startup.sh`. **Do NOT commit `.env`** — transfer securely (SCP, password manager, etc.) when migrating machines.
- **Data files**: `data/`, `models/`, `reports/`, `predictions/*.json` are gitignored. Authoritative state lives in GCS; sync via `gcloud storage rsync gs://progol-data-storage/db data/`.

## Running / testing

```bash
# Local pipeline (interactive prompts)
python run_pipeline.py

# Just the inference path (needs a trained model)
python -m src.progol.ingest.get_progol_ids
python -m src.progol.modeling.predict

# Send Telegram message for the latest concurso
python -m src.progol.bot.send_predictions

# Long-polling bot
python -m src.progol.bot.app

# Tests
pytest                                  # all 88
pytest tests/test_slate_resolution.py   # one file
pytest -k progol_history                # filter
```

`backtest.py` and `train.py` import `joblib` at module top — slim envs without the ML stack will fail to collect tests in those modules. `predict.py` imports joblib **lazily** inside `run()` so its alignment helpers are unit-testable in slim envs.

## External services

| Service | What it's for | How to auth |
|---------|---------------|-------------|
| **API-Football** (`api-sports.io`) | Fixture / odds / injuries data | `FOOTBALL_API_KEY` in `.env` |
| **Google Cloud Storage** (`gs://progol-data-storage`) | DB sync, models, predictions, logs | gcloud ADC; gsutil works, but `google-cloud-storage` Python SDK fails on GCE mTLS on this image — `infra/startup.sh` uses `gsutil rsync` as the reliable path |
| **Telegram Bot API** | Weekly send + interactive `/predecir_*`, `/historial`, etc. | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| **quinielaposible.com** | Scrape current Progol slate (`/category/progol/`) | Public HTML, no auth |
| **GCE VMs** | `progol-trainer` (us-central1-a, STANDARD provisioning — spot preempts truncated runs silently) + `progol-bot` | gcloud SSH |

## Gotchas (learned the hard way)

1. **Slate alignment by original game_number.** When some fixtures fail to resolve, the resolved subset is shorter than 21. Old code used `enumerate(match_ids)` which shifted later picks into the wrong concurso slots (e.g., Nashville's probs ended up on NY Red Bulls' row). Fix lives in `src/progol/modeling/predict.py:align_slots()` + `probs_to_slot_array()`. The `game_numbers` parallel array written by `get_progol_ids.py` carries the original 1..21 slot per match_id; `predict.py` zips on it. Tests: `tests/test_predict_alignment.py`.

2. **Silent partial-write was real.** Concursos 2333 (10/21), 2334 (16/21), and 2335 (10→21 across the May 27 fixes) shipped incomplete because `get_progol_ids.py` couldn't resolve fixtures missing from `NICKNAME_MAP`/`leagues`. The fix is now multi-layered: bigger NICKNAME_MAP (selecciones, MLS, Russia, Chile, J-League, Brazil B, Sweden, Concachampions), bigger league list, threshold 75→70, `days_forward` 5→12. **Adding a new Progol fixture type usually requires: (a) NICKNAME_MAP entry, (b) league ID in `get_progol_ids.py::get_upcoming_api_fixtures()`, and often (c) league ID in `fetch_data.py::LEAGUES` for training data.**

3. **`predict.py` is the actual settlement step.** It calls `database.settle_concurso_actuals()` to backfill `actual_label` from FT matches. `progol_history.py` must run AFTER predict to see settled rows.

4. **Telegram recap on `send_predictions.py` shows the previous concurso's hit-rate** as an appended line. It hides itself if neither bucket has comparable rows (the cn 2326-2330 case where the bot booted after concursos closed).

5. **`compared` vs `settled`** semantics in `database.get_concurso_hits()`: `compared` requires BOTH `predicted_label` and `actual_label`; `settled` only requires the actual. The history table uses `compared` as the hit-rate denominator so "we whiffed everything" looks different from "we never predicted."

6. **Bot deps are slim.** `requirements_bot.txt` excludes the ML stack. `numpy` was needed once `quiniela.optimize_budget` got wired into the bot — keep it in the slim list.

7. **`progol-trainer` VM is STANDARD, not Spot.** Spot preemption silently truncated runs (model trained on partial data, then VM died mid-upload).

8. **GCS uploads use `gsutil rsync` not the Python SDK** on GCE — the metadata-server mTLS handshake breaks `google-auth` in this Debian image. See the comment in `infra/startup.sh` finalize().

9. **Windows workstation SSL.** gcloud uses certifi bundle (sometimes needs `disable_ssl_validation` fallback). `git` uses `http.sslVerify=false`. `pip install` requires `--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org`.

## Conventions

- **Commit messages**: lowercase prefix (`fix:`, `feat:`, `progol-history:`, `batch-a:`), one-line subject, optional bullets.
- **No comments restating what code does.** Only WHY/non-obvious invariants. The codebase generally follows this — match the style of nearby files.
- **Tests live in `tests/`**, mirror the module path roughly. Use `monkeypatch` + `tmp_path` for DB isolation (see `tests/test_progol_history.py::concurso_db`).
- **Concurso schema columns**: `game_number` 1..21, `predicted_label` ∈ {0,1,2,NULL}, `actual_label` ∈ {0,1,2,NULL}. Labels: 0=L(ocal), 1=E(mpate), 2=V(isitante).

## Recently shipped (May 2026)

- Batches A–E (15 of 16): feature engineering, time-decay weights, league-aware Elo, real xG, time-weighted Dirichlet calibration, per-league market blend, drift pressure tracking, per-league dashboard.
- Progol-history feature: `/historial` bot command + automatic last-concurso recap line on weekly Telegram message.
- Slate resolution overhaul (commits `5d1cda9`, `ecfd28f`, `0b2f0ad`): fixed the 10/21 → 21/21 problem for concurso 2335.

## Next on the backlog

- **Healthcheck in `infra/startup.sh`**: after `predict.py`, assert concurso has 21 predicted rows; if not → Telegram alert. (Plan was to do Thursday May 28.)
- **Backfill 2333/2334 actuals** so `/historial` shows their results.
- **Per-slot breakdown in `/historial`** showing which specific picks failed.
- **Investigate Argentina league 128** (per-league acc 0.435 — below average).
- Cup-specific smoothing (Coppa Italia, Coupe de France, DFB-Pokal ECE > 0.13).
- CLV tracking (requires 2 odds snapshots per fixture).
