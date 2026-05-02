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


def _market_probs_from_api(fixture_id: int, session) -> tuple:
    """Returns devigged implied probabilities (sum to 1.0). Training features
    are devigged in preprocess.py, so inference must match — raw 1/odds sums
    to ~1.05-1.10 due to bookmaker overround and would induce a distribution
    shift that drift detection flags on every fixture."""
    try:
        res = session.get(f"{API_BASE}/odds?fixture={fixture_id}&bookmaker=8",
                          timeout=20).json().get('response', [])
        if res and res[0].get('bookmakers'):
            bets = res[0]['bookmakers'][0]['bets'][0]['values']
            raw_h = 1 / float(bets[0]['odd'])
            raw_d = 1 / float(bets[1]['odd'])
            raw_a = 1 / float(bets[2]['odd'])
            overround = raw_h + raw_d + raw_a
            return (raw_h / overround, raw_d / overround, raw_a / overround)
    except Exception as e:
        logger.warning("odds_fetch_failed", extra={'fixture_id': fixture_id, 'err': str(e)})
    return (0.45, 0.25, 0.30)


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


def predict_progol(match_ids):
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
    feature_stats = drift.load_stats(config.FEATURE_STATS_PATH) or {}

    session = api_football_session()
    conn = database.get_connection()
    results = []

    print(f"\nAnalyzing Progol slate ({len(match_ids)} matches) — model {model_version}")

    for mid in match_ids:
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

            market_probs = _market_probs_from_api(mid, session)

            row = build_inference_row(
                conn, home_id=h_id, away_id=a_id,
                league_id=league_id, date=kickoff,
                venue=venue, referee=referee, market_probs=market_probs,
            )

            drift_flags = drift.check_row(row, feature_stats, z_threshold=4.0)
            if drift_flags:
                logger.warning("feature_drift_detected", extra={'fixture_id': mid, 'flags': drift_flags})

            X = pd.DataFrame([row])[feature_cols]
            probs = model.predict_proba(X)[0]
            label = int(np.argmax(probs))

            _log_prediction({
                'fixture_id': mid,
                'predicted_at': datetime.now(timezone.utc).isoformat(),
                'model_version': model_version,
                'home_id': h_id, 'away_id': a_id, 'league_id': league_id, 'kickoff': kickoff,
                'prob_h': float(probs[0]), 'prob_d': float(probs[1]), 'prob_a': float(probs[2]),
                'predicted_label': label, 'drift_flags': drift_flags,
            })

            results.append({'match': f"{h_name} vs {a_name}",
                            'h': probs[0], 'd': probs[1], 'v': probs[2],
                            'drift': bool(drift_flags)})
            logger.info("prediction_ok", extra={'fixture_id': mid, 'label': label})
        except Exception as e:
            logger.error("prediction_failed", extra={'fixture_id': mid, 'err': str(e)})
            continue

    conn.close()

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
        probs_arr = np.array([[r['h'], r['d'], r['v']] for r in results])
        top_n = quiniela.top_n_quinielas(probs_arr, n=10)
        print("TOP-10 QUINIELAS MAS PROBABLES:")
        print(quiniela.format_top_n(top_n))
        print()

        budget_env = os.getenv('PROGOL_BUDGET')
        if budget_env:
            try:
                budget = float(budget_env)
                plan = quiniela.optimize_budget(probs_arr, budget=budget)
                print(f"\nOptimizando con presupuesto = ${budget:.2f} MXN (base ${quiniela.BASE_COST_MXN}):")
                print(quiniela.format_plan(plan))
            except Exception as exc:
                logger.warning(f"budget_optimization_failed: {exc}")

        # Persist for downstream consumers (Telegram bot, dashboards).
        out = {
            'predictions': [{'match': r['match'], 'L': r['h'], 'E': r['d'], 'V': r['v'],
                             'drift': r['drift']} for r in results],
            'top_quinielas': top_n,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'model_version': model_version,
        }
        out_path = config.PROJECT_ROOT / 'predictions' / f'slate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        latest_path = config.PROJECT_ROOT / 'predictions' / 'latest.json'
        latest_path.write_text(json.dumps(out, indent=2))
        gcs.upload_file(out_path, f"predictions/{out_path.name}")
        gcs.upload_file(latest_path, "predictions/latest.json")

    gcs.upload_file(config.PREDICTIONS_DB_PATH, "predictions/predictions.db")


if __name__ == "__main__":
    if config.PROGOL_IDS_PATH.exists():
        with open(config.PROGOL_IDS_PATH, 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
    elif os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
    else:
        logger.error(f"{config.PROGOL_IDS_PATH} not found.")
