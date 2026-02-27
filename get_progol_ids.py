import requests
import re
import json
import os
import logging
from datetime import datetime, timedelta
from thefuzz import process, fuzz
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/match_resolver.log"), logging.StreamHandler()]
)

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
PROGOL_URL = "https://quinielaposible.com/category/progol/"
CACHE_FILE = 'current_progol_ids.json'

def should_skip_scrape():
    """Checks if we already have a valid scrape for today."""
    if not os.path.exists(CACHE_FILE):
        return False
    
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
            # If the cache format is old (just a list), we don't skip
            if isinstance(data, list): return False
            
            last_date = data.get('last_updated')
            if last_date == datetime.now().strftime("%Y-%m-%d"):
                return True
    except:
        return False
    return False

def get_upcoming_api_fixtures():
    headers = {"x-apisports-key": API_KEY}
    seasons = [2025, 2026]
    leagues = [262, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 203, 40, 141, 136, 79, 263, 13, 2, 3]
    all_fixtures = []
    
    logging.info("Fetching fixtures from API to resolve IDs...")
    for lid in leagues:
        for season in seasons:
            try:
                url = f"{BASE_URL}/fixtures?league={lid}&season={season}&next=50"
                res = requests.get(url, headers=headers).json()
                if res.get('response'):
                    all_fixtures.extend(res['response'])
            except: continue
    return all_fixtures

def scrape_progol_matches():
    logging.info(f"Scraping matches from {PROGOL_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(PROGOL_URL, headers=headers)
        text = response.text
        patterns = re.findall(r'([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)\s+vs\s+([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)', text)
        matches = []
        for h, a in patterns:
            h = h.strip().split('\n')[-1].strip()
            a = re.split(r'[\u2013\-\|]', a)[0].strip()
            if len(h) > 1 and len(a) > 1 and "PARTIDO" not in h.upper():
                matches.append((h, a))
                if len(matches) == 14: break
        return matches
    except: return []

def resolve_matches(scraped_matches, api_fixtures):
    resolved_ids = []
    api_map = {f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}": f['fixture']['id'] for f in api_fixtures}
    api_keys = list(api_map.keys())
    
    print("\n--- 🔍 RESOLVING PROGOL IDs ---")
    for i, (home, away) in enumerate(scraped_matches):
        query = f"{home} vs {away}"
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_sort_ratio)
        if score > 70:
            resolved_ids.append(api_map[match])
            print(f"Match {i+1:2}: Resolved ID {api_map[match]} ({score}%)")
    return resolved_ids

if __name__ == "__main__":
    if should_skip_scrape():
        print(f"✅ Progol slate is already up-to-date for today ({datetime.now().strftime('%Y-%m-%d')}). Skipping scrape.")
        exit(0)

    scraped = scrape_progol_matches()
    if not scraped:
        print("Scrape failed."); exit(1)
        
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(scraped, api_data)
    
    # Save with metadata
    cache_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "match_ids": ids
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)
    print(f"\n🚀 Successfully updated Progol IDs for {cache_data['last_updated']}")
