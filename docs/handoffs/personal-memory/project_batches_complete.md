---
name: 16-improvement plan completed (Batches A-E)
description: All 5 batches shipped between 2026-05-15 and 2026-05-17. Concrete deltas vs `baseline-pre-batches` and what was deferred.
type: project
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
Plan executed in 5 batches; 15 of 16 items shipped, 1 deferred. All commits pushed to origin/master. Tag `baseline-pre-batches` at `47c8e47` is the rollback anchor; baseline model preserved at `gs://progol-data-storage/models-baseline/v_20260515_010910/`.

**Concrete metric deltas (baseline → current production model v_20260516_234348):**

| Metric        | baseline | current | Δ |
|---|---|---|---|
| Accuracy (holdout) | 0.5209 | 0.5308 | +0.99pp |
| Log loss (cal)     | 0.9868 | 0.9704 | -1.66%  |
| Brier (cal)        | 0.1962 | 0.1930 | -1.63%  |
| F1-Macro           | 0.3900 | 0.4203 | +0.030  |
| WF avg accuracy    | 0.396  | 0.483  | +8.7pp  |
| ECE (calibrated)   | —      | 0.0131 | new diagnostic |
| Training rows      | 33,627 | 60,190 | +79% (new leagues + cups) |

**What's in production now:**
- Stack: LGB + XGB + CatBoost + RandomForest, isotonic-calibrated, Dirichlet meta-calibration with time-decay weighting.
- Features: 17 numeric + 3 categorical (home_id, referee, league_id), including is_cup, injuries_diff, time-decay-weighted training.
- League coverage: 14 domestic + 10 cups in ingest; 22 leagues in resolver (FA Cup chelsea/MC fix).
- Per-league market blend (predict.py uses config.market_blend_for(league_id)).

**Batch-by-batch:**
- **A** (`4289bb2` + `ca98bcb`): real xG from API expected_goals + injuries diff + alpha_v2_at backfill marker. Full 60k-row backfill ran on trainer ~6h10m.
- **B** (`be342df`): time-decay weighting in train + WF, league-aware Elo cold-start, home_id replaces venue in CAT_COLS, TargetEncoder smoothing 1→10.
- **C** (`e8b013e` reverted by `68ca315`): added PoissonDCEstimator + dropped XGB/RF. Validation showed neutral-to-slightly-worse log_loss (Poisson without per-team params too coarse vs the existing GBMs); reverted but poisson_dc.py removed cleanly.
- **D** (`f9de03b`): time-weighted Dirichlet calibration (sample_weight added to fit_dirichlet), new modeling/metrics.py with ECE + per_league_breakdown, per-league market blend (config.MODEL_MARKET_BLEND_BY_LEAGUE + market_blend_for).
- **E** (`2b739f1`): reporting/league_dashboard.py CLI (--csv writes a snapshot per run), drift pressure tracking (drift.record_run + check_pressure) wired into predict.py with conservative thresholds (3 consecutive runs ≥50% drift).

**Deferred / dropped:**
- **#14 walk-forward ensemble as production model**: requires loading + averaging N model artifacts (~6x size, ~6x inference latency) and the heavy model already trains on fold-6-equivalent data so the ensemble lift is ~0.2-0.5pp at best. Backlog if per-league diagnostics show variance across reruns.
- **CLV (closing line value)**: needs two odds snapshots per fixture (open + close); we only store one. Skipped from Batch D pending an odds-history capture pass.
- **Poisson DC base learner**: full Dixon-Coles MLE with per-team attack/defense (1342×2 params + regularization) would be a proper redo of #2. Current `src/progol/modeling/poisson_dc.py` was removed during the revert; reference design lives in commit `e8b013e` if we revisit.

**How to apply (when looking at future tweaks):**
- Compare any new metric vs the deltas above, not vs the original baseline. The headline number is WF avg accuracy 0.483 — anything below that means we regressed somewhere.
- Per-league dashboard (`python -m src.progol.reporting.league_dashboard --sort ece`) is the right starting point for "what's hurting now". Argentina (128) acc 0.435 and the cups (137 Coppa, 66 Coupe, 81 DFB-Pokal) ECE >0.13 are the worst standing offenders.
- Wednesday 7:07 CDMX scheduler runs the full pipeline including dashboard CSV write. Bot rsyncs predictions/db automatically when /predecir_progol fires.
