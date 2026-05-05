"""Backfill predicted_label / predicted_probs on historical concurso games.

Joins `progol_concurso_games` (rows with fixture_id but no prediction yet)
against `predictions.db` (the per-fixture history written by predict.py
on every run). When a historical fixture happened to be scored in some
prior Progol slate, this copies the latest prediction across.

This is purely a join — it does NOT re-score anything. Concursos that
were never in a live Progol slate stay empty. To produce retroactive
predictions for those, run a separate scoring pass against finished
fixtures with the current model (out of scope here).

Usage:
    python -m src.progol.ingest.backfill_concurso_predictions
    python -m src.progol.ingest.backfill_concurso_predictions --dry-run
"""
import argparse
import json
import logging
import sqlite3

from src.progol.config import DB_PATH, PREDICTIONS_DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Report what would change without writing')
    args = p.parse_args()

    conn_main = sqlite3.connect(DB_PATH)
    conn_main.row_factory = sqlite3.Row
    rows = conn_main.execute('''
        SELECT concurso_number, game_number, fixture_id
        FROM progol_concurso_games
        WHERE fixture_id IS NOT NULL
          AND predicted_probs IS NULL
        ORDER BY concurso_number DESC, game_number
    ''').fetchall()
    logger.info(f"{len(rows)} concurso games eligible (have fixture_id, no prediction yet)")

    if not rows:
        conn_main.close()
        return

    conn_pred = sqlite3.connect(PREDICTIONS_DB_PATH)
    updated = no_pred = 0
    sample_misses = []

    for r in rows:
        cn, gn, fid = r['concurso_number'], r['game_number'], r['fixture_id']
        pred_row = conn_pred.execute('''
            SELECT prob_h, prob_d, prob_a, predicted_label
            FROM predictions
            WHERE fixture_id = ?
            ORDER BY predicted_at DESC LIMIT 1
        ''', (fid,)).fetchone()
        if not pred_row:
            no_pred += 1
            if len(sample_misses) < 5:
                sample_misses.append((cn, gn, fid))
            continue
        ph, pd, pa, label = pred_row
        probs = {'L': float(ph), 'E': float(pd), 'V': float(pa)}
        if not args.dry_run:
            conn_main.execute('''
                UPDATE progol_concurso_games
                SET predicted_label = ?, predicted_probs = ?
                WHERE concurso_number = ? AND game_number = ?
            ''', (int(label), json.dumps(probs), cn, gn))
        updated += 1

    if not args.dry_run:
        conn_main.commit()
    conn_main.close()
    conn_pred.close()
    logger.info(f"updated={updated} no_pred_in_history={no_pred} "
                f"dry_run={args.dry_run}")
    if sample_misses:
        logger.info(f"sample misses: {sample_misses}")


if __name__ == "__main__":
    main()
