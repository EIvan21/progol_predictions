---
name: project-concurso-2333-plan
description: 4-day plan (May 25-28 2026) to diagnose + fix why concurso 2333 shipped with only 15 of 21 predictions
metadata: 
  node_type: memory
  type: project
  originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---

User noticed the Telegram message for concurso 2333 (sent 2026-05-15) only had 15 picks instead of 21 (missing 6 revancha slots). Plan to investigate the week of May 25-28 2026.

**Why:** The pipeline silently shipped a partial concurso — no error surfaced. Two failure modes possible: (a) scrape only resolved 15 of 21 fixtures and `predict.py` wrote what it had, or (b) seeding failed and DB has 0 games for 2333 (confirmed: `progol_concurso_games` has 0 rows for 2333). Latest healthy concurso in DB is 2332 with slot 21 also NULL — same symptom but smaller scale.

**How to apply:** On 2026-05-25 pick up from here. Do NOT start coding until Monday's diagnostic step nails the root cause — both fix paths (scrape vs seed vs alignment) are plausible and the wrong fix would mask the real bug.

## Initial signals already gathered (from 2026-05-22/23 session)

- `predictions/latest.json`: concurso 2333, 15 picks total (14 main + 1 revancha-looking "Atletico Paranaense vs Flamengo"), `generated_at: 2026-05-15`, model `v_20260515_010910`
- DB `progol_concurso_games` rows for 2333: **0**
- DB 2332: 21 games, 20 with predicted_label, **slot 21 (AT. MINEIRO vs BOTAFOGO) = NULL** — same partial-fill smell
- DB 2326–2330: actuals present, no predictions (known historical case, bot booted after concursos closed)
- Stray `slate_20260430_192535.json` with 21 games, all `drift: true`, uniform probs — likely a failed slate that never got cleaned up

## Plan

### Mon 25 — Diagnostic
1. `gsutil rsync` data/ and predictions/ from GCS (DB might be stale vs trainer VM)
2. Check `progol_concursos` table for 2333 (row exists w/o games, or never inserted?)
3. Trainer VM logs: `journalctl -u progol-trainer --since "2026-05-15"` — look for `concurso_seeded`, `fixtures_resolved`, scrape errors
4. Bot VM logs for the 15-may Telegram send
5. Inspect `current_progol_ids.json` vs what Progol site published for 2333
6. **Deliverable:** one-paragraph root cause with log evidence. No code yet.

### Tue 26 — Reproduce + red test
1. Regression test reproducing the 2333 bug (likely new `tests/test_predict_pipeline.py` or extend `test_predict_alignment.py`)
2. Local smoke of `predict.py` against a snapshot of 14-may DB state
3. Branch `fix/concurso-2333-coverage` with red test committed
4. Plan B: if root cause is scrape (not code), `predict.py` should **abort loud** when < 21 fixtures resolved instead of writing partial

### Wed 27 — Fix + live verification
1. Apply fix based on Monday's diagnosis
2. Full `pytest tests/` green
3. Push to master **before** 7:07 CDMX (auto-run on trainer)
4. Watch live: concurso 2334 should ship with 21 games
5. Plan C: rollback + manual Telegram if it breaks
6. **Deliverable:** commit + 2334 with 21 picks + clean Telegram recap

### Thu 28 — Guardrails so this can't fail silently again
1. Healthcheck in `infra/startup.sh`: after predict, assert concurso has 21 predicted games; if not → Telegram alert
2. Backfill 2333: if fixtures finished, run `settle_concurso_actuals(2333)` + check `get_concurso_hits(2333)` even if partial
3. Postmortem memory entry (`project_concurso_2333_postmortem.md`)
4. Stretch: backlog item #3 (per-slot breakdown in `/historial`)

## Files likely touched

- `src/progol/modeling/predict.py` (alignment helpers from [[project-batches-complete]])
- `src/progol/scrape_progol.py` (if scrape is the culprit)
- `src/progol/database.py` (settle_concurso_actuals, seed_concurso)
- `infra/startup.sh` (healthcheck addition)
- `tests/test_predict_alignment.py` or new `tests/test_predict_pipeline.py`

Related: [[project-batches-complete]] for the alignment fix that landed earlier — the helpers exist and are tested, but the silent-partial-write path wasn't covered.
