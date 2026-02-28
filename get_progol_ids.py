import requests
import re
import json
import os
import logging
from datetime import datetime, timedelta
from thefuzz import process, fuzz
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
PROGOL_URL = "https://quinielaposible.com/category/progol/"
CACHE_FILE = 'current_progol_ids.json'

def get_upcoming_api_fixtures(days=10):
    """Fetches a massive list of upcoming fixtures to ensure no match is missed."""
    headers = {"x-apisports-key": API_KEY}
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    
    # We search all major Progol leagues + generic search by date
    leagues = [262, 263, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 203, 40, 141, 136, 79, 13, 2, 3]
    all_fixtures = []
    
    logging.info(f"Searching for all fixtures between {today} and {end_date}...")
    
    # Strategy 1: Search by Date (Broad)
    try:
        url = f"{BASE_URL}/fixtures?from={today}&to={end_date}"
        res = requests.get(url, headers=headers).json()
        if res.get('response'):
            all_fixtures.extend(res['response'])
    except: pass

    # Strategy 2: Search specific leagues if they missed the broad search
    for lid in leagues[:10]:
        try:
            url = f"{BASE_URL}/fixtures?league={lid}&season=2025&next=20"
            res = requests.get(url, headers=headers).json()
            if res.get('response'): all_fixtures.extend(res['response'])
            url_2026 = f"{BASE_URL}/fixtures?league={lid}&season=2026&next=20"
            res_2026 = requests.get(url_2026, headers=headers).json()
            if res_2026.get('response'): all_fixtures.extend(res_2026['response'])
        except: continue
        
    return all_fixtures

def scrape_all_21_matches():
    """Scrapes Progol (14) and Revancha (7)."""
    logging.info(f"Scraping matches from {PROGOL_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(PROGOL_URL, headers=headers)
        text = response.text
        # Regex to find all matches in the page
        patterns = re.findall(r'([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)\s+vs\s+([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)', text)
        
        matches = []
        for h, a in patterns:
            h = h.strip().split('\n')[-1].strip()
            a = re.split(r'[\u2013\-\|]', a)[0].strip()
            if len(h) > 1 and len(a) > 1 and "PARTIDO" not in h.upper():
                matches.append((h, a))
                if len(matches) == 21: break # Total for Progol + Revancha
        return matches
    except: return []

def resolve_matches(scraped_matches, api_fixtures):
    resolved_ids = []
    # Create a giant map of all possible upcoming matches
    api_map = {}
    for f in api_fixtures:
        h = f['teams']['home']['name']
        a = f['teams']['away']['name']
        fid = f['fixture']['id']
        api_map[f"{h} vs {a}"] = fid
        # Add a reverse mapping for neutral venues
        api_map[f"{a} vs {h}"] = fid
        
    api_keys = list(api_map.keys())
    
    print("\n" + "-"*30 + " RESOLVING 21 MATCHES " + "-"*30)
    for i, (home, away) in enumerate(scraped_matches):
        query = f"{home} vs {away}"
        # Use a more tolerant fuzzy match
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_set_ratio)
        
        if score > 65: # Lowered threshold for difficult names
            fid = api_map[match]
            resolved_ids.append(fid)
            section = "PROGOL" if i < 14 else "REVANCHA"
            print(f"{section} {i+1:2}: '{query:25}' -> Found (ID: {fid}, Score: {score})")
        else:
            print(f"FAILED {i+1:2}: '{query:25}' -> No confident match found.")
            
    return resolved_ids

if __name__ == "__main__":
    scraped = scrape_all_21_matches()
    if not scraped:
        print("Scrape failed."); exit(1)
        
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(scraped, api_data)
    
    cache_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "match_ids": ids
    }
    with open(CACHE_FILE, 'w') as f: json.dump(cache_data, f)
    print(f"\n🚀 Successfully resolved {len(ids)} total matches.")
