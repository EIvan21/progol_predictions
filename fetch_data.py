import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import database # Our new database module

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/fetch_data.log"), logging.StreamHandler()]
)

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

# 22 Leagues for a massive dataset
LEAGUES = {
    "Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
    "Ligue 1": 61, "MLS": 253, "Brazil Serie A": 71, "Argentina Primera": 128, 
    "Portugal": 94, "Netherlands": 88, "Belgium": 144, "Turkey": 203,
    "Championship": 40, "Spain Segunda": 141, "Italy Serie B": 136, "Germany 2. Bundesliga": 79,
    "Liga MX Expansion": 263, "Copa Libertadores": 13, "Champions League": 2, "Europa League": 3,
    "Eredivisie": 88
}

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

def fetch_league_data_incremental(league_id, season, name):
    """Only fetches data AFTER the latest match we have in DB."""
    last_date = database.get_latest_match_date(league_id, season)
    
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": API_KEY}
    
    # Define date parameters if we have previous data
    params = {"league": league_id, "season": season}
    if last_date:
        # Fetch from the day after the last match
        start_date = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        params["from"] = start_date
        params["to"] = datetime.now().strftime("%Y-%m-%d")
        
        # Check if start_date is in the future
        if start_date > params["to"]:
            logging.info(f"Skipping {name} {season}: Database is already up to date.")
            return

    try:
        logging.info(f"Requesting {name} (ID: {league_id}) {season} from {params.get('from', 'Beginning')}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        matches = data.get('response', [])
        if not matches:
            logging.info(f"No new matches found for {name} {season}.")
            return
            
        # Save to SQLite instead of JSON files
        saved_count = database.save_matches_to_db(matches, season)
        logging.info(f"SUCCESS: Saved {saved_count} new matches to DB for {name} {season}.")
        
    except Exception as e:
        logging.error(f"Error fetching {name} {season}: {e}")

if __name__ == "__main__":
    if not API_KEY:
        logging.critical("FOOTBALL_API_KEY missing in .env!")
    else:
        database.init_db()
        logging.info("Starting Incremental Data Ingestion...")
        for name, league_id in LEAGUES.items():
            for season in SEASONS:
                fetch_league_data_incremental(league_id, season, name)
                time.sleep(1.2) # Rate limit
        logging.info("Incremental Ingestion Task Completed.")
