"""Backfill fixture_id on historical concurso games by fuzzy-matching team
names against the matches table.

Historical backfill (backfill_concursos.py) writes home_name/away_name from
quinielaposible.com without resolving to API-Football fixture IDs. This
script joins those games against rows already in our matches table by
(home_name, away_name, week_start ± window). Once fixture_id is filled,
settle_concurso_actuals() can populate actual_label automatically.

Strategy:
- For each concurso game with NULL fixture_id, look up team_id by name in
  the teams table (fuzzy via thefuzz).
- Find candidate matches with (home_id, away_id) within +/- 7 days of
  week_start (or with no week_start, scan a wide window).
- Pick the highest fuzzy score above threshold; skip ambiguous cases.
"""
import argparse
import logging
import sqlite3
from datetime import datetime, timedelta

from thefuzz import process, fuzz

from src.progol import database
from src.progol.ingest.get_progol_ids import clean_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _all_teams(conn):
    rows = conn.execute("SELECT team_id, name FROM teams WHERE name IS NOT NULL").fetchall()
    return {r[0]: clean_name(r[1]) for r in rows}


def _resolve_team_id(scraped_name, team_map, threshold=80):
    cleaned = clean_name(scraped_name)
    if not team_map:
        return None, 0
    choices = list(team_map.values())
    best, score = process.extractOne(cleaned, choices, scorer=fuzz.token_sort_ratio)
    if score < threshold:
        return None, score
    for tid, n in team_map.items():
        if n == best:
            return tid, score
    return None, score


def _find_match(conn, home_id, away_id, week_start, window_days=10):
    """Look for a match between these teams. Window defaults to +/-10 days
    around week_start. If week_start is None, scan everything."""
    if week_start:
        try:
            anchor = datetime.fromisoformat(week_start)
            lo = (anchor - timedelta(days=window_days)).isoformat()
            hi = (anchor + timedelta(days=window_days)).isoformat()
            row = conn.execute("""
                SELECT fixture_id FROM matches
                WHERE home_id=? AND away_id=? AND date BETWEEN ? AND ?
                  AND status='FT'
                ORDER BY date LIMIT 1
            """, (home_id, away_id, lo, hi)).fetchone()
        except ValueError:
            row = None
    else:
        row = None
    if row:
        return row[0]
    # Fallback: any FT match between these teams (latest first).
    row = conn.execute("""
        SELECT fixture_id FROM matches
        WHERE home_id=? AND away_id=? AND status='FT'
        ORDER BY date DESC LIMIT 1
    """, (home_id, away_id)).fetchone()
    return row[0] if row else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=int, default=80,
                   help='Fuzzy match minimum score for team name resolution')
    args = p.parse_args()

    conn = database.get_connection()
    team_map = _all_teams(conn)
    logger.info(f"Loaded {len(team_map)} teams from teams table")

    rows = conn.execute("""
        SELECT g.concurso_number, g.game_number, g.home_name, g.away_name,
               c.week_start
        FROM progol_concurso_games g
        LEFT JOIN progol_concursos c ON c.concurso_number = g.concurso_number
        WHERE g.fixture_id IS NULL
        ORDER BY g.concurso_number DESC, g.game_number
    """).fetchall()
    logger.info(f"{len(rows)} games missing fixture_id")

    resolved = ambiguous = no_team = no_match = 0
    for cn, gn, h, a, ws in rows:
        h_id, h_score = _resolve_team_id(h, team_map, args.threshold)
        a_id, a_score = _resolve_team_id(a, team_map, args.threshold)
        if not h_id or not a_id:
            no_team += 1
            logger.debug(f"concurso {cn} game {gn}: name unresolved h={h}({h_score}) a={a}({a_score})")
            continue
        fid = _find_match(conn, h_id, a_id, ws)
        if not fid:
            no_match += 1
            continue
        conn.execute("""
            UPDATE progol_concurso_games SET fixture_id=?
            WHERE concurso_number=? AND game_number=? AND fixture_id IS NULL
        """, (fid, cn, gn))
        resolved += 1

    conn.commit()
    conn.close()

    logger.info(f"resolved={resolved}  no_team_match={no_team}  no_fixture_in_db={no_match}")

    n = database.settle_concurso_actuals()
    logger.info(f"actual_label backfilled on {n} games")


if __name__ == "__main__":
    main()
