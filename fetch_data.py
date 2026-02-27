import os
import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/fetch_data.log"), logging.StreamHandler()]
)

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}

LEAGUES = {"Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "MLS": 253}
SEASONS = [2023, 2024]

def fetch_match_details(fixture_id):
    try:
        url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
        res = requests.get(url, headers=headers).json()
        if not res.get('response'): return None
        
        stats = {}
        # Team 0 is usually Home, Team 1 is Away
        for i, team_stat in enumerate(res['response']):
            prefix = 'home' if i == 0 else 'away'
            for item in team_stat['statistics']:
                val = item['value']
                if item['type'] == 'Shots on Goal': stats[f'{prefix}_shots'] = int(val or 0)
                if item['type'] == 'Ball Possession': stats[f'{prefix}_possession'] = int(str(val or "0").replace('%',''))
                if item['type'] == 'Corner Kicks': stats[f'{prefix}_corners'] = int(val or 0)
        return stats
    except: return None

def enrich_database_fully():
    """Loops until ALL matches have statistics."""
    while True:
        conn = database.get_connection()
        # Fetch a batch of 50 to process (Smaller batches are safer for rate limits)
        query = "SELECT fixture_id FROM matches WHERE status = 'FT' AND home_shots IS NULL LIMIT 50"
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        
        if not fixtures:
            logging.info("🎉 SUCCESS: Entire database enriched with detailed statistics.")
            break

        logging.info(f"Processing Batch: {len(fixtures)} matches remaining...")
        for fid in fixtures:
            details = fetch_match_details(fid)
            if details:
                database.update_match_stats(fid, details)
                logging.info(f"✅ Enriched Fixture {fid}")
            else:
                # Mark as 'Processed with 0' to avoid infinite loops if API has no data
                database.update_match_stats(fid, {'home_shots': 0})
            
            time.sleep(1.5) # Protect your API Key from being banned

if __name__ == "__main__":
    database.init_db()
    
    # 1. Standard Fetch
    logging.info("Standard incremental fetch starting...")
    for name, lid in LEAGUES.items():
        for season in SEASONS:
            last_date = database.get_latest_match_date(lid, season)
            params = {"league": lid, "season": season}
            if last_date:
                start = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                params["from"], params["to"] = start, datetime.now().strftime("%Y-%m-%d")
                if start > params["to"]: continue
            
            res = requests.get(f"{BASE_URL}/fixtures", headers=headers, params=params).json()
            database.save_matches_to_db(res.get('response', []), season)
            time.sleep(1.2)

    # 2. FULL Enrichment (No more 100 limit!)
    enrich_database_fully()
