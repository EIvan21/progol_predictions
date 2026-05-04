import json
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progol_concursos (
            concurso_number INTEGER PRIMARY KEY,
            week_start TEXT,
            source_url TEXT,
            scraped_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progol_concurso_games (
            concurso_number INTEGER,
            game_number INTEGER,
            home_name TEXT,
            away_name TEXT,
            fixture_id INTEGER,
            predicted_label INTEGER,
            predicted_probs TEXT,
            actual_label INTEGER,
            settled_at TEXT,
            PRIMARY KEY (concurso_number, game_number),
            FOREIGN KEY (concurso_number) REFERENCES progol_concursos(concurso_number)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_concurso_games_fixture ON progol_concurso_games(fixture_id)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_users (
            chat_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            role TEXT NOT NULL DEFAULT 'pending',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            last_seen_at TEXT
        )
    ''')
    # Migration for DBs created before user_id existed. SQLite raises
    # OperationalError on duplicate add — swallow it so init_db stays
    # idempotent.
    try:
        cursor.execute("ALTER TABLE bot_users ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_users_user_id ON bot_users(user_id)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_threads (
            thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            state_json TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            started_at TEXT NOT NULL,
            closed_at TEXT,
            FOREIGN KEY (chat_id) REFERENCES bot_users(chat_id)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_threads_chat ON bot_threads(chat_id, status)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            thread_id INTEGER,
            direction TEXT NOT NULL,
            command TEXT,
            text TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_messages_chat ON bot_messages(chat_id, created_at)")
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


def upsert_concurso(concurso_number, week_start, source_url, games):
    """Persist a Progol concurso (header + games). `games` is a list of dicts:
    {game_number, home_name, away_name, fixture_id (or None)}."""
    if not concurso_number or not games:
        return
    conn = get_connection()
    conn.execute('''
        INSERT INTO progol_concursos (concurso_number, week_start, source_url, scraped_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(concurso_number) DO UPDATE SET
            week_start=COALESCE(excluded.week_start, week_start),
            source_url=excluded.source_url,
            scraped_at=excluded.scraped_at
    ''', (concurso_number, week_start, source_url))
    for g in games:
        # ON CONFLICT: preserve predicted_label/actual_label across re-scrapes.
        # COALESCE on fixture_id so the historical backfill (which passes None)
        # doesn't null a fixture_id resolved earlier by the live scraper.
        conn.execute('''
            INSERT INTO progol_concurso_games
                (concurso_number, game_number, home_name, away_name, fixture_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(concurso_number, game_number) DO UPDATE SET
                home_name=excluded.home_name, away_name=excluded.away_name,
                fixture_id=COALESCE(excluded.fixture_id, fixture_id)
        ''', (concurso_number, g['game_number'], g['home_name'], g['away_name'], g.get('fixture_id')))
    conn.commit()
    conn.close()


def update_concurso_prediction(concurso_number, game_number, predicted_label, predicted_probs):
    """Called from predict.py once the model has scored a fixture."""
    conn = get_connection()
    conn.execute('''
        UPDATE progol_concurso_games
        SET predicted_label = ?, predicted_probs = ?
        WHERE concurso_number = ? AND game_number = ?
    ''', (predicted_label, predicted_probs, concurso_number, game_number))
    conn.commit()
    conn.close()


def settle_concurso_actuals():
    """Backfill actual_label from finished matches. Idempotent."""
    conn = get_connection()
    conn.execute('''
        UPDATE progol_concurso_games
        SET actual_label = (
            SELECT CASE
                WHEN m.goals_home > m.goals_away THEN 0
                WHEN m.goals_home = m.goals_away THEN 1
                ELSE 2
            END
            FROM matches m
            WHERE m.fixture_id = progol_concurso_games.fixture_id
              AND m.status = 'FT'
        ),
        settled_at = datetime('now')
        WHERE actual_label IS NULL
          AND fixture_id IN (SELECT fixture_id FROM matches WHERE status = 'FT')
    ''')
    n = conn.total_changes
    conn.commit()
    conn.close()
    return n


def get_concurso_with_games(concurso_number):
    """Returns (header_dict, games_list) for the bot/dashboard."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    header = conn.execute(
        "SELECT * FROM progol_concursos WHERE concurso_number = ?", (concurso_number,)
    ).fetchone()
    games = conn.execute(
        "SELECT * FROM progol_concurso_games WHERE concurso_number = ? ORDER BY game_number",
        (concurso_number,),
    ).fetchall()
    conn.close()
    return (dict(header) if header else None, [dict(g) for g in games])


def get_latest_concurso_number():
    conn = get_connection()
    cur = conn.execute("SELECT MAX(concurso_number) FROM progol_concursos")
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


# --- Bot auth + conversation persistence ---------------------------------

def bot_get_user(chat_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM bot_users WHERE chat_id=?", (chat_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def bot_upsert_user(chat_id, user_id=None, username=None, first_name=None,
                    last_name=None, role=None, status=None):
    """Insert as 'pending' on first contact; refresh profile + last_seen_at on
    every subsequent message. role/status are only changed when explicitly
    passed (otherwise preserved across calls)."""
    conn = get_connection()
    conn.execute('''
        INSERT INTO bot_users (chat_id, user_id, username, first_name, last_name,
                               role, status, created_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, COALESCE(?, 'pending'), COALESCE(?, 'pending'),
                datetime('now'), datetime('now'))
        ON CONFLICT(chat_id) DO UPDATE SET
            user_id=COALESCE(excluded.user_id, user_id),
            username=COALESCE(excluded.username, username),
            first_name=COALESCE(excluded.first_name, first_name),
            last_name=COALESCE(excluded.last_name, last_name),
            role=COALESCE(?, role),
            status=COALESCE(?, status),
            last_seen_at=datetime('now')
    ''', (chat_id, user_id, username, first_name, last_name, role, status,
          role, status))
    conn.commit()
    conn.close()


def bot_set_role(chat_id, role, status=None):
    conn = get_connection()
    if status is not None:
        cur = conn.execute(
            "UPDATE bot_users SET role=?, status=? WHERE chat_id=?",
            (role, status, chat_id),
        )
    else:
        cur = conn.execute(
            "UPDATE bot_users SET role=? WHERE chat_id=?",
            (role, chat_id),
        )
    n = cur.rowcount
    conn.commit()
    conn.close()
    return n


def bot_list_users():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM bot_users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def bot_is_authorized(chat_id, allowed_roles=('owner', 'admin', 'user')):
    user = bot_get_user(chat_id)
    if not user:
        return False
    return user['status'] == 'active' and user['role'] in allowed_roles


def bot_log_message(chat_id, direction, text, command=None, thread_id=None):
    conn = get_connection()
    conn.execute('''
        INSERT INTO bot_messages (chat_id, thread_id, direction, command, text, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    ''', (chat_id, thread_id, direction, command, text))
    conn.commit()
    conn.close()


def bot_open_thread(chat_id, topic, state=None):
    state_json = json.dumps(state) if state is not None else None
    conn = get_connection()
    cur = conn.execute('''
        INSERT INTO bot_threads (chat_id, topic, state_json, status, started_at)
        VALUES (?, ?, ?, 'open', datetime('now'))
    ''', (chat_id, topic, state_json))
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return tid


def bot_get_open_thread(chat_id, topic):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute('''
        SELECT * FROM bot_threads
        WHERE chat_id=? AND topic=? AND status='open'
        ORDER BY started_at DESC LIMIT 1
    ''', (chat_id, topic)).fetchone()
    conn.close()
    return dict(row) if row else None


def bot_set_thread_state(thread_id, state):
    conn = get_connection()
    conn.execute(
        "UPDATE bot_threads SET state_json=? WHERE thread_id=?",
        (json.dumps(state), thread_id),
    )
    conn.commit()
    conn.close()


def bot_close_thread(thread_id):
    conn = get_connection()
    conn.execute(
        "UPDATE bot_threads SET status='closed', closed_at=datetime('now') WHERE thread_id=?",
        (thread_id,),
    )
    conn.commit()
    conn.close()
