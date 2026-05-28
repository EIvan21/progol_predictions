# Session transcript — 2026-05-28

The full raw conversation is committed alongside this file as **`2026-05-28-transcript.jsonl`** (~15 MB). It is the authoritative record; this markdown is just a human-readable summary.

To get a nicer formatted markdown of the same content, run `/export` from inside Claude Code on either machine — that command can only be invoked interactively from the CLI; I cannot run it from a tool call.

## Session summary (manually written)

This session spanned roughly a week (May 22–28, 2026) with several discrete pieces of work. In rough order:

### Earlier (recapped from prior compaction)
- Built the **progol-history feature**: `database.get_concurso_hits()` + `reporting/progol_history.py` CLI + `/historial` bot command, all wired into `infra/startup.sh`.
- Added the **weekly recap line** to `bot/send_predictions.py` so the Telegram message shows last concurso's hit-rate.
- Added **31 tests** total (alignment + progol_history).

### Mid-session (May 22)
- User noticed concurso 2333 only shipped 15 picks instead of 21.
- Initial diagnostic: `latest.json` had 15 picks, DB had 0 rows for 2333 (later corrected — GCS copy had 21/21 because trainer re-ran twice on May 16).
- Created `project_concurso_2333_plan.md` in personal memory — a 4-day plan (Mon 25 – Thu 28).

### Mon 25 → skipped (user worked on other things)

### Tue 26 — Diagnostic + first fix wave (commit `5d1cda9`)
- Downloaded trainer logs from GCS for May 15 and May 20.
- Found cn 2333 was 15/21 resolved, cn 2334 was 16/21. Common failures: missing `NICKNAME_MAP` entries (MLS short names, Russia, Chile, J-League, women's) + missing leagues + strict fuzzy threshold.
- Fix: expanded `NICKNAME_MAP` ~+22 entries, added Chile/J1/J2/UWCL to leagues, threshold 75→70, `days_forward` 5→8.
- Added `tests/test_slate_resolution.py` (34 tests).
- Pull on bot VM.

### Wed 27 — Manual trainer re-run reveals more failures (commit `ecfd28f`)
- User asked to re-run trainer to predict cn 2335. First run: 10/21 resolved.
- Discovered cn 2335 had 7 international friendlies (selecciones nacionales in Spanish) plus Brazil Serie B + Swedish Allsvenskan — none mapped.
- Fix: added ~50 national-team Spanish↔English entries, leagues 10 (Friendlies), 72 (Brazil B), 113/114 (Sweden), widened `days_forward` 8→12.
- Second trainer run: 20/21 resolved (TOLUCA vs TIGRES still failed).

### Wed 27 evening — CONCACAF final fix (commit `0b2f0ad`)
- User clarified TOLUCA vs TIGRES is the Concachampions final, not Liga MX.
- Fix: added league 16 (CONCACAF Champions Cup) to both `get_progol_ids.py` and `fetch_data.py`.
- Third trainer run: **21/21 resolved**. Telegram sent OK.

### Thu 28 — This handoff session
- User migrating to a new machine. Generated CLAUDE.md, this handoff doc, bundled personal memory, documented environment, committing everything.

## Stats

| Metric | Before week | After week |
|--------|-------------|------------|
| Tests | 31 | 88 |
| `NICKNAME_MAP` entries | ~30 | ~110 |
| API leagues fetched | 29 | 37 |
| Worst recent concurso resolution | 10/21 | 21/21 |
| Commits this week | 0 | 3 (`5d1cda9`, `ecfd28f`, `0b2f0ad`) + this handoff |

## Files to inspect first when picking back up

- `src/progol/ingest/get_progol_ids.py` — the central piece, especially `NICKNAME_MAP`, `get_upcoming_api_fixtures::leagues`, `resolve_matches`.
- `src/progol/modeling/predict.py:align_slots, probs_to_slot_array` — alignment helpers that prevent slot-shift bugs.
- `infra/startup.sh` — full pipeline; the **healthcheck addition** is the next planned safeguard.
- `src/progol/bot/send_predictions.py:60-67` — the recap append (defensive try/except wrap).
- `src/progol/database.py` (functions `get_concurso_hits`, `settle_concurso_actuals`, `get_previous_concurso_number`).
- `tests/test_slate_resolution.py`, `tests/test_predict_alignment.py`, `tests/test_progol_history.py` — newest tests.
