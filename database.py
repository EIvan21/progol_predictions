import sqlite3
import pandas as pd
import os
import logging

DB_PATH = "data/progol.db"

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Matches table
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
            UNIQUE(fixture_id)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")

def save_matches_to_db(matches_list, season):
    """Inserts or replaces matches in the database."""
    if not matches_list:
        return 0
        
    conn = get_connection()
    count = 0
    for m in matches_list:
        try:
            conn.execute('''
                INSERT OR REPLACE INTO matches 
                (fixture_id, league_id, season, date, venue, referee, home_id, home_name, away_id, away_name, goals_home, goals_away, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                m['fixture']['id'],
                m['league']['id'],
                season,
                m['fixture']['date'],
                m['fixture']['venue']['name'] or "Unknown",
                m['fixture']['referee'] or "Unknown",
                m['teams']['home']['id'],
                m['teams']['home']['name'],
                m['teams']['away']['id'],
                m['teams']['away']['name'],
                m['goals']['home'],
                m['goals']['away'],
                m['fixture']['status']['short']
            ))
            count += 1
        except Exception as e:
            continue
            
    conn.commit()
    conn.close()
    return count

def get_latest_match_date(league_id, season):
    """Returns the date of the most recent match for a league/season."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM matches WHERE league_id = ? AND season = ?", (league_id, season))
    res = cursor.fetchone()[0]
    conn.close()
    return res

def get_all_matches_df():
    """Returns all matches as a Pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM matches WHERE status = 'FT'", conn)
    conn.close()
    return df

def get_h2h_stats(team1_id, team2_id, current_date):
    """Calculates historical H2H results between two teams before a specific date."""
    conn = get_connection()
    query = '''
        SELECT goals_home, goals_away, home_id 
        FROM matches 
        WHERE status = 'FT' 
        AND date < ? 
        AND ((home_id = ? AND away_id = ?) OR (home_id = ? AND away_id = ?))
    '''
    df = pd.read_sql_query(query, conn, params=(current_date, team1_id, team2_id, team2_id, team1_id))
    conn.close()
    
    if df.empty:
        return 0.33, 0.33, 0.33 # Equal probability if no history
        
    def get_res(row):
        if row['goals_home'] == row['goals_away']: return 'D'
        if row['home_id'] == team1_id:
            return 'W' if row['goals_home'] > row['goals_away'] else 'L'
        else:
            return 'W' if row['goals_away'] > row['goals_home'] else 'L'
            
    results = df.apply(get_res, axis=1).value_counts(normalize=True)
    return results.get('W', 0), results.get('D', 0), results.get('L', 0)
