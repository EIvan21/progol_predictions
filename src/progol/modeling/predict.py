import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.progol import config, database
from src.progol.features.team_state import build_inference_row
from src.progol.modeling import quiniela
from src.progol.utils import drift
from src.progol.utils.http import api_football_session
from src.progol.utils.logging_setup import configure as configure_logging
from src.progol.storage import versioning, gcs

load_dotenv()
logger = logging.getLogger(__name__)

API_BASE = "https://v3.football.api-sports.io"


def _injuries_from_api(fixture_id: int, home_id: int, away_id: int, session):
    """Pull current /injuries for the fixture and split by side. Returns
    (h_inj, a_inj). Best-effort: any failure -> (0, 0) so the feature
    stays defined. For training rows this same logic ran in fetch_data
    so train/inference symmetry is preserved (0 on API miss, not None)."""
    try:
        res = session.get(f"{API_BASE}/injuries?fixture={fixture_id}", timeout=15).json().get('response', [])
        h, a = 0, 0
        for item in res:
            tid = (item.get('team') or {}).get('id')
            if tid == home_id:
                h += 1
            elif tid == away_id:
                a += 1
        return h, a
    except Exception:
        return 0, 0


def _market_probs_from_api(fixture_id: int, session):
    """Returns (probs, has_market). probs sums to 1.0 — devigged from bookmaker
    overround. has_market=False means the API returned nothing and probs is the
    generic 1X2 prior; callers should NOT blend that into the model output."""
    try:
        res = session.get(f"{API_BASE}/odds?fixture={fixture_id}&bookmaker=8",
                          timeout=20).json().get('response', [])
        if res and res[0].get('bookmakers'):
            bets = res[0]['bookmakers'][0]['bets'][0]['values']
            raw_h = 1 / float(bets[0]['odd'])
            raw_d = 1 / float(bets[1]['odd'])
            raw_a = 1 / float(bets[2]['odd'])
            overround = raw_h + raw_d + raw_a
            return (raw_h / overround, raw_d / overround, raw_a / overround), True
    except Exception as e:
        logger.warning("odds_fetch_failed", extra={'fixture_id': fixture_id, 'err': str(e)})
    return (0.45, 0.25, 0.30), False


def _init_predictions_db():
    conn = sqlite3.connect(config.PREDICTIONS_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER, predicted_at TEXT, model_version TEXT,
            home_id INTEGER, away_id INTEGER, league_id INTEGER, kickoff TEXT,
            prob_h REAL, prob_d REAL, prob_a REAL, predicted_label INTEGER,
            actual_label INTEGER, drift_flags TEXT,
            UNIQUE(fixture_id, model_version)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_fixture ON predictions(fixture_id)")
    conn.commit()
    conn.close()


def _log_prediction(row: dict):
    conn = sqlite3.connect(config.PREDICTIONS_DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO predictions
        (fixture_id, predicted_at, model_version, home_id, away_id, league_id, kickoff,
         prob_h, prob_d, prob_a, predicted_label, drift_flags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['fixture_id'], row['predicted_at'], row['model_version'],
        row['home_id'], row['away_id'], row['league_id'], row['kickoff'],
        row['prob_h'], row['prob_d'], row['prob_a'], row['predicted_label'],
        json.dumps(row.get('drift_flags', {})),
    ))
    conn.commit()
    conn.close()


def _apply_post_calibration(probs, temperature, dirichlet_cal):
    """Reusable post-calibration: prefer Dirichlet (per-class) when fitted,
    fall back to scalar temperature, else identity. `probs` is a 1-D vector
    or a 2-D batch."""
    p = np.asarray(probs)
    single = p.ndim == 1
    if single:
        p = p[np.newaxis, :]
    if dirichlet_cal is not None:
        from src.progol.modeling.train import apply_dirichlet
        p = apply_dirichlet(p, dirichlet_cal)
    elif abs(temperature - 1.0) > 1e-3:
        eps = 1e-12
        clipped = np.clip(p, eps, 1.0)
        scaled = np.exp(np.log(clipped) / temperature)
        p = scaled / scaled.sum(axis=1, keepdims=True)
    return p[0] if single else p


def predict_upcoming_fixtures(model, feature_cols, model_version, temperature,
                              feature_stats, days_ahead=7,
                              dirichlet_cal=None):
    """Score every fixture in `matches` with status != 'FT' whose kickoff is
    within `days_ahead` days. Persists to `match_predictions` for the bot to
    serve `/predecir_partido` for non-Progol matches.

    Skips bookmaker odds (would be N extra API-Football calls per run); the
    bot tags these as model-only via the dedicated table. Temperature is
    applied for parity with the Progol path."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    hi = now + timedelta(days=days_ahead)

    database.cleanup_old_match_predictions(stale_after_days=1)

    # Need a session for the /injuries call below — kept here (not at module
    # scope) so unit tests can monkeypatch without a network reach.
    session = api_football_session()

    conn = database.get_connection()
    rows = conn.execute("""
        SELECT m.fixture_id, m.league_id, m.date, m.home_id, m.away_id,
               m.venue, m.referee, h.name as home_name, a.name as away_name
        FROM matches m
        LEFT JOIN teams h ON h.team_id = m.home_id
        LEFT JOIN teams a ON a.team_id = m.away_id
        WHERE COALESCE(m.status, '') != 'FT'
          AND m.date >= ? AND m.date < ?
        ORDER BY m.date
    """, (now.isoformat(), hi.isoformat())).fetchall()

    written = failed = 0
    for fid, league_id, date, h_id, a_id, venue, referee, h_name, a_name in rows:
        try:
            h_inj, a_inj = _injuries_from_api(fid, h_id, a_id, session)
            row = build_inference_row(
                conn, home_id=h_id, away_id=a_id,
                league_id=league_id, date=date,
                venue=venue or "Unknown",
                referee=referee or "Unknown",
                market_probs=(0.45, 0.25, 0.30),
                home_injuries=h_inj, away_injuries=a_inj,
            )
            X = pd.DataFrame([row])[feature_cols]
            model_probs = model.predict_proba(X)[0]
            model_probs = _apply_post_calibration(model_probs, temperature, dirichlet_cal)
            label = int(np.argmax(model_probs))
            database.upsert_match_prediction(
                fixture_id=fid, league_id=league_id, date=date,
                home_id=h_id, away_id=a_id,
                home_name=h_name or '', away_name=a_name or '',
                predicted_label=label,
                predicted_probs={
                    'L': float(model_probs[0]),
                    'E': float(model_probs[1]),
                    'V': float(model_probs[2]),
                },
                model_version=model_version,
            )
            written += 1
        except Exception as exc:
            failed += 1
            logger.warning(f"upcoming_predict_failed fixture={fid}: {exc}")
    conn.close()
    logger.info("upcoming_predictions_persisted",
                extra={'written': written, 'failed': failed,
                       'window_days': days_ahead})
    print(f"\nUpcoming fixtures predicted: {written} (failed {failed}, window {days_ahead}d)")
    return written


def predict_progol(match_ids, slate_meta=None, game_numbers=None):
    """`game_numbers` (optional) is a list parallel to `match_ids` carrying
    each fixture's original Progol slot (1..21). When some scraped matches
    fail to resolve, the resolved set has gaps in slot numbers; without
    this mapping the i-th prediction silently lands at concurso row=i,
    pairing the right L/E/V with the wrong team labels. If omitted we fall
    back to 1..N to preserve the old behavior."""
    configure_logging()
    _init_predictions_db()

    model_path = versioning.resolve_latest(config.MODEL_DIR) or config.PRIMARY_MODEL_PATH
    if isinstance(model_path, type(config.MODEL_DIR)) and model_path.is_dir():
        model_path = model_path / 'calibrated_ensemble.pkl'
    if not model_path.exists():
        logger.error("Model not found. Run training first.")
        return

    pkg = joblib.load(model_path)
    model = pkg['model']
    feature_cols = pkg['features']
    model_version = pkg.get('version', 'unversioned')
    temperature = float(pkg.get('temperature', 1.0))
    dirichlet_cal = pkg.get('dirichlet_calibrator')
    feature_stats = drift.load_stats(config.FEATURE_STATS_PATH) or {}

    session = api_football_session()
    conn = database.get_connection()
    results = []

    # Optional global override (env) — if set, applies to ALL matches and
    # bypasses the per-league map. Useful for ablation testing ("how does
    # the model do if I trust the market 100%?"). When unset (the normal
    # production path), each fixture's blend is looked up by league via
    # config.market_blend_for() inside the per-fixture loop below.
    blend_override = os.getenv('MODEL_MARKET_BLEND')
    blend_w_global = (min(max(float(blend_override), 0.0), 1.0)
                      if blend_override is not None else None)

    # The slate file's concurso_number lets us link each prediction back to the
    # progol_concurso_games row; game_number is the 1-indexed position in match_ids.
    concurso_number = slate_meta.get('concurso_number') if slate_meta else None

    cal_label = pkg.get('calibration', 'temperature' if abs(temperature - 1.0) > 1e-3 else 'none')
    blend_desc = (f"{blend_w_global:.2f} (override)" if blend_w_global is not None
                  else "per-league")
    print(f"\nAnalyzing Progol slate ({len(match_ids)} matches) — model {model_version} (cal={cal_label}, T={temperature:.3f}, blend={blend_desc})")
    if concurso_number:
        print(f"Concurso: {concurso_number}")

    # Align each fixture to its ORIGINAL Progol slot (1..21). If the caller
    # didn't pass game_numbers (older slate JSON), fall back to enumeration.
    if game_numbers and len(game_numbers) == len(match_ids):
        slot_pairs = list(zip(game_numbers, match_ids))
    else:
        slot_pairs = list(enumerate(match_ids, start=1))

    for game_idx, mid in slot_pairs:
        try:
            res = session.get(f"{API_BASE}/fixtures?id={mid}", timeout=20).json()
            if not res.get('response'):
                continue
            m = res['response'][0]
            h_name = m['teams']['home']['name']
            a_name = m['teams']['away']['name']
            h_id = m['teams']['home']['id']
            a_id = m['teams']['away']['id']
            league_id = m['league']['id']
            kickoff = m['fixture']['date']
            venue = (m['fixture']['venue'] or {}).get('name') or "Unknown"
            referee = m['fixture'].get('referee') or "Unknown"

            market_probs, has_market = _market_probs_from_api(mid, session)
            h_inj, a_inj = _injuries_from_api(mid, h_id, a_id, session)

            row = build_inference_row(
                conn, home_id=h_id, away_id=a_id,
                league_id=league_id, date=kickoff,
                venue=venue, referee=referee, market_probs=market_probs,
                home_injuries=h_inj, away_injuries=a_inj,
            )

            drift_flags = drift.check_row(row, feature_stats, z_threshold=4.0)
            if drift_flags:
                logger.warning("feature_drift_detected", extra={'fixture_id': mid, 'flags': drift_flags})

            X = pd.DataFrame([row])[feature_cols]
            model_probs = model.predict_proba(X)[0]

            # Post-calibration (Dirichlet preferred, temperature fallback)
            # is applied before the market blend so the blend weight
            # operates on properly-calibrated model output.
            model_probs = _apply_post_calibration(model_probs, temperature, dirichlet_cal)

            if has_market:
                # Per-league blend weight (Batch D). Falls back to default
                # if the league isn't in the override map.
                blend_w = (blend_w_global if blend_w_global is not None
                           else config.market_blend_for(league_id))
                blended = blend_w * model_probs + (1.0 - blend_w) * np.array(market_probs)
                probs = blended / blended.sum()
            else:
                probs = model_probs
            label = int(np.argmax(probs))

            _log_prediction({
                'fixture_id': mid,
                'predicted_at': datetime.now(timezone.utc).isoformat(),
                'model_version': model_version,
                'home_id': h_id, 'away_id': a_id, 'league_id': league_id, 'kickoff': kickoff,
                'prob_h': float(probs[0]), 'prob_d': float(probs[1]), 'prob_a': float(probs[2]),
                'predicted_label': label, 'drift_flags': drift_flags,
            })

            if concurso_number:
                database.update_concurso_prediction(
                    concurso_number, game_idx, label,
                    json.dumps({'L': float(probs[0]), 'E': float(probs[1]), 'V': float(probs[2])}),
                )

            results.append({'match': f"{h_name} vs {a_name}",
                            'h': probs[0], 'd': probs[1], 'v': probs[2],
                            'drift': bool(drift_flags),
                            'gn': int(game_idx)})
            logger.info("prediction_ok", extra={'fixture_id': mid, 'label': label})
        except Exception as e:
            logger.error("prediction_failed", extra={'fixture_id': mid, 'err': str(e)})
            continue

    conn.close()

    # Backfill actual_label for any concurso games whose fixtures have settled.
    try:
        n = database.settle_concurso_actuals()
        if n:
            logger.info("concurso_actuals_backfilled", extra={'rows': n})
    except Exception as exc:
        logger.warning(f"settle_concurso_actuals failed: {exc}")

    print("\n" + "=" * 40 + " PROGOL REPORT " + "=" * 40)
    print(f"{'GAME':<3} | {'MATCHUP':<35} | {'L %':<6} | {'E %':<6} | {'V %':<6} | PRED | DRIFT")
    print("-" * 105)
    for i, r in enumerate(results):
        idx = int(np.argmax([r['h'], r['d'], r['v']]))
        label = {0: 'L', 1: 'E', 2: 'V'}[idx]
        d = "*" if r['drift'] else ""
        print(f"{i+1:<3} | {r['match']:<35} | {r['h']*100:5.1f}% | {r['d']*100:5.1f}% | {r['v']*100:5.1f}% |  {label}    |  {d}")
    print("=" * 105 + "\n")

    if results:
        # Index probs by ORIGINAL slot so the plan / top quinielas line up
        # with concurso game_numbers 1..14, not the resolved-list position.
        # Unresolved main-slate slots get a neutral 1X2 prior so the budget
        # optimizer still considers all 14 positions (those slots will tend
        # to surface as triples since their distribution is flat-ish).
        probs_by_slot = {r['gn']: [r['h'], r['d'], r['v']] for r in results}
        DEFAULT_PRIOR = [0.45, 0.25, 0.30]
        main_probs = [probs_by_slot.get(s, DEFAULT_PRIOR) for s in range(1, 15)]
        probs_arr = np.array(main_probs)

        # MC sampling instead of greedy top-N: greedy returns 10 quinielas
        # that differ from the modal pick by 1-2 swaps; MC produces
        # genuinely diverse alternates weighted by P(L,E,V).
        top_n = quiniela.mc_sample_quinielas(probs_arr, n=10)
        print("TOP-10 QUINIELAS (Monte Carlo):")
        print(quiniela.format_top_n(top_n))
        print()

        # Always compute a default budget plan so the Telegram bot has
        # something concrete to show in place of the (now-removed)
        # top-quinielas table. PROGOL_BUDGET env overrides the default.
        budget = float(os.getenv('PROGOL_BUDGET', 300))
        plan = None
        try:
            plan = quiniela.optimize_budget(probs_arr, budget=budget)
            print(f"\nPlan optimizado con presupuesto = ${budget:.2f} MXN (base ${quiniela.BASE_COST_MXN}):")
            print(quiniela.format_plan(plan))
        except Exception as exc:
            logger.warning(f"budget_optimization_failed: {exc}")

        plan_dict = None
        if plan is not None:
            plan_dict = {
                'budget': budget,
                'cost': plan.cost,
                'coverage_prob': plan.coverage_prob,
                'n_tickets': plan.n_tickets,
                'doubles': list(plan.doubles),
                'triples': list(plan.triples),
                'base_picks': list(plan.base_picks),
                'matches_summary': plan.matches_summary,
            }

        # Persist for downstream consumers (Telegram bot, dashboards).
        out = {
            'concurso_number': concurso_number,
            'predictions': [{'match': r['match'], 'L': r['h'], 'E': r['d'], 'V': r['v'],
                             'drift': r['drift']} for r in results],
            'top_quinielas': top_n,
            'plan': plan_dict,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'model_version': model_version,
        }
        out_path = config.PROJECT_ROOT / 'predictions' / f'slate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        latest_path = config.PROJECT_ROOT / 'predictions' / 'latest.json'
        latest_path.write_text(json.dumps(out, indent=2))

    # Pre-compute predictions for arbitrary upcoming fixtures BEFORE any
    # gcs.upload_file calls. Those uploads sometimes fail on this image's
    # mTLS metadata server and would otherwise abort the upcoming pass;
    # finalize's `gsutil rsync` handles uploads regardless.
    try:
        predict_upcoming_fixtures(
            model=model, feature_cols=feature_cols,
            model_version=model_version, temperature=temperature,
            feature_stats=feature_stats, days_ahead=7,
            dirichlet_cal=dirichlet_cal,
        )
    except Exception as exc:
        logger.warning(f"upcoming_predict_pass_failed: {exc}")

    # GCS uploads via the Python SDK are best-effort — finalize's
    # `gsutil rsync` reliably picks up everything in predictions/ + the
    # log dir. Wrap each call so a single failure can't take down the
    # rest of the pipeline (or upcoming-fixture pass above).
    if results:
        try:
            gcs.upload_file(out_path, f"predictions/{out_path.name}")
        except Exception as exc:
            logger.warning(f"gcs_upload_slate_failed: {exc}")
        try:
            gcs.upload_file(latest_path, "predictions/latest.json")
        except Exception as exc:
            logger.warning(f"gcs_upload_latest_failed: {exc}")
    try:
        gcs.upload_file(config.PREDICTIONS_DB_PATH, "predictions/predictions.db")
    except Exception as exc:
        logger.warning(f"gcs_upload_predictions_db_failed: {exc}")


if __name__ == "__main__":
    slate_path = config.PROGOL_IDS_PATH if config.PROGOL_IDS_PATH.exists() else None
    if slate_path is None and os.path.exists('current_progol_ids.json'):
        slate_path = 'current_progol_ids.json'
    if slate_path is None:
        logger.error(f"{config.PROGOL_IDS_PATH} not found.")
    else:
        with open(slate_path, 'r') as f:
            slate_meta = json.load(f)
        predict_progol(
            slate_meta.get('match_ids', []),
            slate_meta=slate_meta,
            game_numbers=slate_meta.get('game_numbers'),
        )
