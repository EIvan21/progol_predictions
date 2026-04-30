import sqlite3
import pandas as pd

def check_health():
    conn = sqlite3.connect("data/progol.db")
    df = pd.read_sql_query("SELECT league_id, season, COUNT(*) as match_count FROM matches GROUP BY league_id, season", conn)
    
    league_map = {262: "Liga MX", 39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1", 253: "MLS"}
    df['league_name'] = df['league_id'].map(league_map).fillna(df['league_id'].astype(str))
    
    print("
--- 📊 DATABASE HEALTH REPORT ---")
    print(df[['league_name', 'season', 'match_count']].to_string(index=False))
    print(f"
TOTAL UNIQUE MATCHES: {df['match_count'].sum():,}")
    conn.close()

if __name__ == "__main__":
    check_health()
