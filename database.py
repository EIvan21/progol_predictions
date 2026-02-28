import sqlite3
import pandas as pd
import os
import logging

DB_PATH = "data/progol.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection(); cursor = conn.cursor()
    # Expanded schema for Alpha Strategy
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
            
            -- Dimension 3: Market Signal
            odds_home FLOAT,
            odds_draw FLOAT,
            odds_away FLOAT,
            odds_movement FLOAT,
            
            -- Dimension 4: xG Engine
            home_xg FLOAT,
            away_xg FLOAT,
            
            UNIQUE(fixture_id)
        )
    ''')
    conn.commit(); conn.close()
    logging.info("Database initialized with Alpha Signal schema.")

def save_matches_to_db(matches_list, season):
    if not matches_list: return 0
    conn = get_connection()
    count = 0
    for m in matches_list:
        try:
            conn.execute('INSERT OR REPLACE INTO matches (fixture_id, league_id, season, date, venue, referee, home_id, away_id, goals_home, goals_away, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (m['fixture']['id'], m['league']['id'], season, m['fixture']['date'], m['fixture']['venue']['name'] or "Unknown", m['fixture']['referee'] or "Unknown", m['teams']['home']['id'], m['teams']['away']['id'], m['goals']['home'], m['goals']['away'], m['fixture']['status']['short']))
            count += 1
        except: continue
    conn.commit(); conn.close()
    return count

def update_alpha_stats(fixture_id, data):
    """Updates match with Odds and xG."""
    conn = get_connection()
    conn.execute('''
        UPDATE matches SET 
        home_shots = ?, away_shots = ?, home_possession = ?, away_possession = ?, 
        home_corners = ?, away_corners = ?, 
        odds_home = ?, odds_draw = ?, odds_away = ?,
        home_xg = ?, away_xg = ?
        WHERE fixture_id = ?
    ''', (
        data.get('h_sh'), data.get('a_sh'), data.get('h_po'), data.get('a_po'),
        data.get('h_co'), data.get('a_co'),
        data.get('o_h'), data.get('o_d'), data.get('o_a'),
        data.get('h_xg'), data.get('a_xg'),
        fixture_id
    ))
    conn.commit(); conn.close()

def get_latest_match_date(league_id, season):
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM matches WHERE league_id = ? AND season = ?", (league_id, season))
    res = cursor.fetchone()[0]; conn.close()
    return res

def get_all_matches_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'FT'", conn)
    conn.close(); return df
