import requests
import re
from bs4 import BeautifulSoup
from thefuzz import process, fuzz
import json
import os
import logging
from datetime import datetime, timedelta
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

# All supported leagues from fetch_data.py
SEARCH_LEAGUES = [262, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 203, 40, 141, 136, 79, 263, 13, 2, 3]

def get_upcoming_api_fixtures(days=7):
    headers = {"x-apisports-key": API_KEY}
    today = datetime.now()
    all_fixtures = []
    
    # Season to check: both current and previous (for 2026 crossover)
    seasons = [2025, 2026]
    
    logging.info(f"Searching fixtures for Seasons {seasons} in {len(SEARCH_LEAGUES)} leagues...")
    
    # To be efficient and avoid hitting limits, we fetch the next 50 fixtures for each league
    for lid in SEARCH_LEAGUES:
        for season in seasons:
            try:
                url = f"{BASE_URL}/fixtures?league={lid}&season={season}&next=50"
                res = requests.get(url, headers=headers).json()
                if res.get('response'):
                    all_fixtures.extend(res['response'])
            except Exception as e:
                logging.error(f"Error fetching League {lid} Season {season}: {e}")
                continue
    
    logging.info(f"Total fixtures collected for matching: {len(all_fixtures)}")
    return all_fixtures

def scrape_progol_matches():
    try:
        logging.info(f"Scraping matches from {PROGOL_URL}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(PROGOL_URL, headers=headers)
        response.raise_for_status()
        
        matches = []
        text = response.text
        patterns = re.findall(r'([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)\s+vs\s+([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)', text)
        
        for h, a in patterns:
            h = h.strip().split('\n')[-1].strip()
            a = re.split(r'[\u2013\-\|]', a)[0].strip()
            if len(h) > 1 and len(a) > 1 and "PARTIDO" not in h.upper():
                matches.append((h, a))
                if len(matches) == 14: break
        
        logging.info(f"Scraped {len(matches)} matches.")
        return matches
    except Exception as e:
        logging.error(f"Scraping Error: {e}")
        return []

def resolve_matches(scraped_matches, api_fixtures):
    resolved_ids = []
    api_map = {}
    for f in api_fixtures:
        h = f['teams']['home']['name']
        a = f['teams']['away']['name']
        fid = f['fixture']['id']
        api_map[f"{h} vs {a}"] = fid
        
    api_keys = list(api_map.keys())
    
    print("\n--- RESOLVING MATCHES ---")
    if not api_keys:
        print("Error: No API fixtures available for matching.")
        return []

    for i, (home, away) in enumerate(scraped_matches):
        query = f"{home} vs {away}"
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_sort_ratio)
        
        if score > 70:
            fid = api_map[match]
            resolved_ids.append(fid)
            print(f"Match {i+1:2}: '{query:25}' -> '{match:25}' (ID: {fid}, Score: {score})")
        else:
            print(f"Match {i+1:2}: '{query:25}' -> No confident match (Best: '{match}', Score: {score})")
            
    return resolved_ids

if __name__ == "__main__":
    scraped = scrape_progol_matches()
    if not scraped:
        print("Failed to scrape matches.")
        exit(1)
        
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(scraped, api_data)
    
    if ids:
        with open('current_progol_ids.json', 'w') as f:
            json.dump(ids, f)
        print(f"\nSaved {len(ids)} resolved match IDs to 'current_progol_ids.json'")
    else:
        print("\nNo matches were resolved.")
