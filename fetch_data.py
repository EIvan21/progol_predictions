import os
import json
import requests
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

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

# 6 Seasons: 2019 to 2024
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

def save_locally(json_data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(json_data, f)

def fetch_league_data(league_id, season, name):
    url = f"{BASE_URL}/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": league_id, "season": season}
    
    try:
        logging.info(f"Requesting {name} (ID: {league_id}) for Season {season}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('response'):
            logging.warning(f"No data found for {name} {season}.")
            return
            
        filename = f"data/raw/fixtures_{league_id}_{season}.json"
        save_locally(data, filename)
        logging.info(f"Successfully saved {len(data['response'])} matches for {name} {season}.")
        
    except Exception as e:
        logging.error(f"Error fetching {name} {season}: {e}")

if __name__ == "__main__":
    if not API_KEY:
        logging.critical("FOOTBALL_API_KEY missing in .env!")
    else:
        logging.info("Starting Massive Data Ingestion...")
        for name, league_id in LEAGUES.items():
            for season in SEASONS:
                fetch_league_data(league_id, season, name)
                # Rate limit safety (adjust if you have a higher tier)
                time.sleep(1.2) 
        logging.info("Ingestion Task Completed.")
