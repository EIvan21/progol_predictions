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

def fetch_alpha_details(fid):
    """Fetches Stats and Odds simultaneously."""
    try:
        # 1. Stats Call
        s_res = requests.get(f"{BASE_URL}/fixtures/statistics?fixture={fid}", headers=headers).json().get('response', [])
        stats = {}
        if s_res:
            for i, ts in enumerate(s_res):
                p = 'h' if i == 0 else 'a'
                s_map = {item['type']: item['value'] for item in ts['statistics']}
                stats[f'{p}_sh'] = int(s_map.get('Shots on Goal', 0) or 0)
                stats[f'{p}_po'] = int(str(s_map.get('Ball Possession', "0") or "0").replace('%',''))
                stats[f'{p}_co'] = int(s_map.get('Corner Kicks', 0) or 0)
                # Calculate xG Proxy: (Shots on Goal * 0.3) + (Total Shots * 0.1)
                total_sh = int(s_map.get('Total Shots', 0) or 0)
                stats[f'{p}_xg'] = (stats[f'{p}_sh'] * 0.3) + (total_sh * 0.1)

        # 2. Odds Call (Bookmaker 8 is usually stable historical data)
        o_res = requests.get(f"{BASE_URL}/odds?fixture={fid}&bookmaker=8", headers=headers).json().get('response', [])
        if o_res and o_res[0].get('bookmakers'):
            bets = o_res[0]['bookmakers'][0]['bets'][0]['values']
            stats['o_h'] = float(bets[0]['odd'])
            stats['o_d'] = float(bets[1]['odd'])
            stats['o_a'] = float(bets[2]['odd'])
            
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
                else: database.update_alpha_stats(fid, {'o_h': 0}) # Mark processed
        time.sleep(0.5)

if __name__ == "__main__":
    database.init_db()
    # 1. Fetching logic here...
    # 2. Start Turbo Alpha Ingestion
    enrich_database_alpha()
