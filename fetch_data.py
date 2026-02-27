import os
import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/fetch_data.log"), logging.StreamHandler()]
)

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}

def fetch_match_details(fixture_id):
    try:
        url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
        # We don't use timeout here to avoid skipping slow but valid responses
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
    """Max throughput parallel fetching."""
    while True:
        conn = database.get_connection()
        # Larger batch for efficiency
        query = "SELECT fixture_id FROM matches WHERE status = 'FT' AND home_shots IS NULL LIMIT 200"
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        
        if not fixtures:
            logging.info("🎉 SUCCESS: All matches enriched.")
            break

        logging.info(f"⚡ TURBO PROCESSING: {len(fixtures)} matches in batch...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fid = {executor.submit(fetch_match_details, fid): fid for fid in fixtures}
            
            for future in as_completed(future_to_fid):
                fid, stats = future.result()
                if stats:
                    database.update_match_stats(fid, stats)
                else:
                    # Mark as 0 so we don't try it again today
                    database.update_match_stats(fid, {'home_shots': 0})
        
        # Short pause between batches to allow DB to breathe
        time.sleep(0.5)

if __name__ == "__main__":
    database.init_db()
    
    # Standard check for new fixtures first
    # (Omitted league loop for brevity here, logic remains same)
    
    # Start the Turbo Enrichment
    enrich_database_turbo(max_workers=10)
