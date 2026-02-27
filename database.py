import sqlite3
import pandas as pd
import os
import logging

DB_PATH = "data/progol.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Expanded Matches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            fixture_id INTEGER PRIMARY KEY,
            league_id INTEGER,
            season INTEGER,
            date TEXT,
            venue TEXT,
            referee TEXT,
            home_id INTEGER,
            home_name TEXT,
            away_id INTEGER,
            away_name TEXT,
            goals_home INTEGER,
            goals_away INTEGER,
            status TEXT,
            
            -- Dimension 1: Match Statistics
            home_shots INTEGER,
            away_shots INTEGER,
            home_possession INTEGER,
            away_possession INTEGER,
            home_corners INTEGER,
            away_corners INTEGER,
            
            -- Dimension 2: Standings
            home_rank INTEGER,
            away_rank INTEGER,
            
            UNIQUE(fixture_id)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database expanded with Stats and Standings.")

def save_matches_to_db(matches_list, season):
    if not matches_list: return 0
    conn = get_connection()
    count = 0
    for m in matches_list:
        try:
            conn.execute('''
                INSERT OR REPLACE INTO matches 
                (fixture_id, league_id, season, date, venue, referee, home_id, home_name, away_id, away_name, goals_home, goals_away, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                m['fixture']['id'], m['league']['id'], season, m['fixture']['date'],
                m['fixture']['venue']['name'] or "Unknown", m['fixture']['referee'] or "Unknown",
                m['teams']['home']['id'], m['teams']['home']['name'],
                m['teams']['away']['id'], m['teams']['away']['name'],
                m['goals']['home'], m['goals']['away'], m['fixture']['status']['short']
            ))
            count += 1
        except: continue
    conn.commit()
    conn.close()
    return count

def update_match_stats(fixture_id, stats_data):
    """Updates a specific match with its detailed statistics."""
    conn = get_connection()
    # stats_data example: {'home_shots': 10, 'away_shots': 5, ...}
    conn.execute('''
        UPDATE matches SET 
        home_shots = ?, away_shots = ?, 
        home_possession = ?, away_possession = ?, 
        home_corners = ?, away_corners = ?
        WHERE fixture_id = ?
    ''', (
        stats_data.get('home_shots'), stats_data.get('away_shots'),
        stats_data.get('home_possession'), stats_data.get('away_possession'),
        stats_data.get('home_corners'), stats_data.get('away_corners'),
        fixture_id
    ))
    conn.commit()
    conn.close()

def update_match_ranks(fixture_id, home_rank, away_rank):
    conn = get_connection()
    conn.execute('UPDATE matches SET home_rank = ?, away_rank = ? WHERE fixture_id = ?', (home_rank, away_rank, fixture_id))
    conn.commit()
    conn.close()

def get_latest_match_date(league_id, season):
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM matches WHERE league_id = ? AND season = ?", (league_id, season))
    res = cursor.fetchone()[0]
    conn.close()
    return res

def get_all_matches_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'FT'", conn)
    conn.close()
    return df
