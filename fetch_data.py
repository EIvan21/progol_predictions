import os
import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}

LEAGUES = {
    "Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
    "Ligue 1": 61, "MLS": 253, "Brazil Serie A": 71, "Argentina": 128, "Portugal": 94,
    "Championship": 40, "Eredivisie": 88, "Liga MX Expansion": 263
}
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

def fetch_alpha_details(fid):
    try:
        s_res = requests.get(f"{BASE_URL}/fixtures/statistics?fixture={fid}", headers=headers).json().get('response', [])
        stats = {}
        if s_res:
            for i, ts in enumerate(s_res):
                p = 'h' if i == 0 else 'a'
                s_map = {item['type']: item['value'] for item in ts['statistics']}
                stats[f'{p}_sh'] = int(s_map.get('Shots on Goal', 0) or 0)
                stats[f'{p}_po'] = int(str(s_map.get('Ball Possession', "0") or "0").replace('%',''))
                stats[f'{p}_co'] = int(s_map.get('Corner Kicks', 0) or 0)
                total_sh = int(s_map.get('Total Shots', 0) or 0)
                stats[f'{p}_xg'] = (stats[f'{p}_sh'] * 0.3) + (total_sh * 0.1)

        o_res = requests.get(f"{BASE_URL}/odds?fixture={fid}&bookmaker=8", headers=headers).json().get('response', [])
        if o_res and o_res[0].get('bookmakers'):
            bets = o_res[0]['bookmakers'][0]['bets'][0]['values']
            stats['o_h'], stats['o_d'], stats['o_a'] = float(bets[0]['odd']), float(bets[1]['odd']), float(bets[2]['odd'])
        return fid, stats
    except: return fid, None

def enrich_database_alpha(max_workers=10):
    while True:
        conn = database.get_connection()
        query = "SELECT fixture_id FROM matches WHERE status = 'FT' AND odds_home IS NULL LIMIT 100"
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        if not fixtures: break
        logging.info(f"⚡ Alpha Enrichment: {len(fixtures)} matches in batch...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fid = {executor.submit(fetch_alpha_details, fid): fid for fid in fixtures}
            for future in as_completed(future_to_fid):
                fid, data = future.result()
                if data: database.update_alpha_stats(fid, data)
                else: database.update_alpha_stats(fid, {'o_h': 0})
        time.sleep(0.5)

if __name__ == "__main__":
    database.init_db()
    # RESTORED: Fixture Discovery Logic
    logging.info("Step 1: Discovering Matches (2019-2026)...")
    for name, lid in LEAGUES.items():
        for season in SEASONS:
            last_date = database.get_latest_match_date(lid, season)
            params = {"league": lid, "season": season}
            if last_date:
                start = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                params["from"], params["to"] = start, datetime.now().strftime("%Y-%m-%d")
                if start > params["to"]: continue
            
            logging.info(f"Fetching {name} {season} fixtures...")
            try:
                res = requests.get(f"{BASE_URL}/fixtures", headers=headers, params=params).json()
                matches = res.get('response', [])
                if matches:
                    database.save_matches_to_db(matches, season)
                    logging.info(f"✅ Added {len(matches)} matches.")
                time.sleep(1.2)
            except: continue

    # Step 2: Turbo Alpha Enrichment
    enrich_database_alpha()
