import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/fetch_data.log"), logging.StreamHandler()]
)

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
FETCH_CACHE_FILE = 'data/last_fetch.json'

LEAGUES = {
    "Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
    "Ligue 1": 61, "MLS": 253, "Brazil Serie A": 71, "Argentina Primera": 128, 
    "Portugal": 94, "Netherlands": 88, "Belgium": 144, "Turkey": 203,
    "Championship": 40, "Spain Segunda": 141, "Italy Serie B": 136, "Germany 2. Bundesliga": 79,
    "Liga MX Expansion": 263, "Copa Libertadores": 13, "Champions League": 2, "Europa League": 3,
    "Eredivisie": 88
}

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

def should_skip_fetch():
    """Checks if we have already attempted an API fetch today."""
    if not os.path.exists(FETCH_CACHE_FILE):
        return False
    try:
        with open(FETCH_CACHE_FILE, 'r') as f:
            cache = json.load(f)
            return cache.get('last_fetch_date') == datetime.now().strftime("%Y-%m-%d")
    except:
        return False

def record_fetch_attempt():
    """Records that we successfully checked for data today."""
    os.makedirs('data', exist_ok=True)
    with open(FETCH_CACHE_FILE, 'w') as f:
        json.dump({'last_fetch_date': datetime.now().strftime("%Y-%m-%d")}, f)

def fetch_league_data_incremental(league_id, season, name):
    last_date = database.get_latest_match_date(league_id, season)
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": league_id, "season": season}
    
    if last_date:
        start_date = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        params["from"] = start_date
        params["to"] = datetime.now().strftime("%Y-%m-%d")
        if start_date > params["to"]: return

    try:
        logging.info(f"Checking {name} {season} for new matches...")
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        matches = data.get('response', [])
        if matches:
            saved = database.save_matches_to_db(matches, season)
            logging.info(f"Added {saved} new matches for {name}.")
    except Exception as e:
        logging.error(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    if not API_KEY:
        logging.critical("API Key missing!")
        exit(1)

    if should_skip_fetch():
        print(f"\n✅ DATABASE ALREADY UPDATED TODAY ({datetime.now().strftime('%Y-%m-%d')}). skipping API search.")
        exit(0)

    database.init_db()
    logging.info("Starting Daily Data Check...")
    for name, league_id in LEAGUES.items():
        for season in SEASONS:
            fetch_league_data_incremental(league_id, season, name)
            time.sleep(1.2)
    
    record_fetch_attempt()
    logging.info("Daily Data Check Completed.")
