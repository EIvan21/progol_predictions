import sqlite3
import pandas as pd
import logging

from src.progol.config import DB_PATH, DATA_DIR


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            fixture_id INTEGER PRIMARY KEY,
            league_id INTEGER,
            season INTEGER,
            date TEXT,
            venue TEXT,
            referee TEXT,
            home_id INTEGER,
            away_id INTEGER,
            goals_home INTEGER,
            goals_away INTEGER,
            status TEXT,
            home_shots INTEGER,
            away_shots INTEGER,
            home_possession INTEGER,
            away_possession INTEGER,
            home_corners INTEGER,
            away_corners INTEGER,
            odds_home FLOAT,
            odds_draw FLOAT,
            odds_away FLOAT,
            odds_movement FLOAT,
            home_xg FLOAT,
            away_xg FLOAT,
            home_rank INTEGER,
            away_rank INTEGER,
            home_form TEXT,
            away_form TEXT,
            venue_id INTEGER,
            venue_surface TEXT,
            h2h_home_wins INTEGER,
            h2h_draws INTEGER,
            h2h_away_wins INTEGER,
            UNIQUE(fixture_id)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_id, date)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT,
            code TEXT,
            founded INTEGER,
            venue_id INTEGER,
            venue_name TEXT,
            venue_surface TEXT,
            logo TEXT,
            updated_at TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized with Alpha Signal schema.")


def upsert_team(team_id, payload):
    """Insert or update a team row. payload is the API-Football /teams item."""
    if not payload:
        return
    team = payload.get('team', {}) or {}
    venue = payload.get('venue', {}) or {}
    conn = get_connection()
    conn.execute('''
        INSERT INTO teams (team_id, name, country, code, founded, venue_id,
                           venue_name, venue_surface, logo, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(team_id) DO UPDATE SET
            name=excluded.name, country=excluded.country, code=excluded.code,
            founded=excluded.founded, venue_id=excluded.venue_id,
            venue_name=excluded.venue_name, venue_surface=excluded.venue_surface,
            logo=excluded.logo, updated_at=excluded.updated_at
    ''', (
        team_id, team.get('name'), team.get('country'), team.get('code'),
        team.get('founded'), venue.get('id'), venue.get('name'),
        venue.get('surface'), team.get('logo'),
    ))
    conn.commit()
    conn.close()


def get_team_name(team_id):
    conn = get_connection()
    cur = conn.execute("SELECT name FROM teams WHERE team_id = ?", (team_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def save_matches_to_db(matches_list, season):
    if not matches_list:
        return 0
    conn = get_connection()
    count = 0
    for m in matches_list:
        try:
            conn.execute(
                'INSERT OR REPLACE INTO matches (fixture_id, league_id, season, date, venue, referee, home_id, away_id, goals_home, goals_away, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    m['fixture']['id'], m['league']['id'], season, m['fixture']['date'],
                    m['fixture']['venue']['name'] or "Unknown",
                    m['fixture']['referee'] or "Unknown",
                    m['teams']['home']['id'], m['teams']['away']['id'],
                    m['goals']['home'], m['goals']['away'],
                    m['fixture']['status']['short'],
                ),
            )
            count += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return count


def update_alpha_stats(fixture_id, data):
    conn = get_connection()
    conn.execute('''
        UPDATE matches SET
        home_shots = ?, away_shots = ?, home_possession = ?, away_possession = ?,
        home_corners = ?, away_corners = ?,
        odds_home = ?, odds_draw = ?, odds_away = ?,
        home_xg = ?, away_xg = ?,
        home_rank = ?, away_rank = ?,
        home_form = ?, away_form = ?,
        venue_id = ?, venue_surface = ?,
        h2h_home_wins = ?, h2h_draws = ?, h2h_away_wins = ?
        WHERE fixture_id = ?
    ''', (
        data.get('h_sh'), data.get('a_sh'), data.get('h_po'), data.get('a_po'),
        data.get('h_co'), data.get('a_co'),
        data.get('o_h'), data.get('o_d'), data.get('o_a'),
        data.get('h_xg'), data.get('a_xg'),
        data.get('h_rank'), data.get('a_rank'),
        data.get('h_form'), data.get('a_form'),
        data.get('v_id'), data.get('v_surf'),
        data.get('h2h_h'), data.get('h2h_d'), data.get('h2h_a'),
        fixture_id,
    ))
    conn.commit()
    conn.close()


def get_latest_match_date(league_id, season):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM matches WHERE league_id = ? AND season = ?", (league_id, season))
    res = cursor.fetchone()[0]
    conn.close()
    return res


def get_all_matches_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'FT'", conn)
    conn.close()
    return df
