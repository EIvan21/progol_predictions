import os
import json
import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}
FETCH_CACHE_FILE = 'data/last_fetch.json'

# Restore Full League List and Seasons
LEAGUES = {
    "Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
    "Ligue 1": 61, "MLS": 253, "Brazil Serie A": 71, "Argentina": 128, "Portugal": 94,
    "Championship": 40, "Eredivisie": 88, "Liga MX Expansion": 263
}
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

def should_skip_fetch():
    if not os.path.exists(FETCH_CACHE_FILE): return False
    try:
        with open(FETCH_CACHE_FILE, 'r') as f:
            cache = json.load(f)
            return cache.get('last_fetch_date') == datetime.now().strftime("%Y-%m-%d")
    except: return False

def record_fetch_attempt():
    os.makedirs('data', exist_ok=True)
    with open(FETCH_CACHE_FILE, 'w') as f:
        json.dump({'last_fetch_date': datetime.now().strftime("%Y-%m-%d")}, f)

def fetch_match_details(fixture_id):
    try:
        url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
        res = requests.get(url, headers=headers).json()
        if not res.get('response'): return fixture_id, None
        stats = {}
        for i, team_stat in enumerate(res['response']):
            prefix = 'home' if i == 0 else 'away'
            for item in team_stat['statistics']:
                val = item['value']
                if item['type'] == 'Shots on Goal': stats[f'{prefix}_shots'] = int(val or 0)
                if item['type'] == 'Ball Possession': stats[f'{prefix}_possession'] = int(str(val or "0").replace('%',''))
                if item['type'] == 'Corner Kicks': stats[f'{prefix}_corners'] = int(val or 0)
        return fixture_id, stats
    except: return fixture_id, None

def enrich_database_turbo(max_workers=10):
    while True:
        conn = database.get_connection()
        query = "SELECT fixture_id FROM matches WHERE status = 'FT' AND home_shots IS NULL LIMIT 200"
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        if not fixtures: break
        logging.info(f"⚡ TURBO PROCESSING: {len(fixtures)} matches in batch...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fid = {executor.submit(fetch_match_details, fid): fid for fid in fixtures}
            for future in as_completed(future_to_fid):
                fid, stats = future.result()
                if stats: database.update_match_stats(fid, stats)
                else: database.update_match_stats(fid, {'home_shots': 0})
        time.sleep(0.5)

if __name__ == "__main__":
    if should_skip_fetch():
        print(f"\n✅ DATABASE ALREADY UPDATED TODAY ({datetime.now().strftime('%Y-%m-%d')}). skipping API search.")
        exit(0)

    database.init_db()
    logging.info("Starting Fresh Data Fetch (Restoring 2019-2024 History)...")
    for name, lid in LEAGUES.items():
        for season in SEASONS:
            last_date = database.get_latest_match_date(lid, season)
            params = {"league": lid, "season": season}
            if last_date:
                start = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                params["from"], params["to"] = start, datetime.now().strftime("%Y-%m-%d")
                if start > params["to"]: continue
            
            logging.info(f"Fetching {name} {season}...")
            res = requests.get(f"{BASE_URL}/fixtures", headers=headers, params=params).json()
            matches = res.get('response', [])
            if matches: database.save_matches_to_db(matches, season)
            time.sleep(1.2)
    
    enrich_database_turbo()
    record_fetch_attempt()
    logging.info("Daily Fetch Completed and Cached.")
