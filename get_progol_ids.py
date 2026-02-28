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

def clean_name(name):
    """Aggressively cleans team names for better fuzzy matching."""
    name = name.upper()
    # Remove common suffixes/prefixes
    removals = ["FC", "CF", "CD", "UD", "CLUB", "DEPORTIVO", "REAL", "ATLETICO", "AT.", "S.C.", "U.N.A.M.", "UANL"]
    for r in removals:
        name = name.replace(r, "").strip()
    return name

def get_upcoming_api_fixtures(days=14): # Increased to 14 days
    headers = {"x-apisports-key": API_KEY}
    start = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    
    leagues = [262, 263, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 203, 40, 141, 136, 79, 13, 2, 3]
    all_fixtures = []
    
    # 1. Broad Search
    try:
        res = requests.get(f"{BASE_URL}/fixtures?from={start}&to={end}", headers=headers).json()
        if res.get('response'): all_fixtures.extend(res['response'])
    except: pass

    # 2. Season-Specific Deep Search (2025 and 2026)
    for lid in leagues:
        for season in [2025, 2026]:
            try:
                res = requests.get(f"{BASE_URL}/fixtures?league={lid}&season={season}&next=50", headers=headers).json()
                if res.get('response'): all_fixtures.extend(res['response'])
            except: continue
    return all_fixtures

def scrape_full_slate():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(PROGOL_URL, headers=headers)
        text = response.text
        # Look for the 'vs' pattern across the whole page
        patterns = re.findall(r'([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)\s+vs\s+([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)', text)
        
        matches = []
        for h, a in patterns:
            h = h.strip().split('\n')[-1].strip()
            a = re.split(r'[\u2013\-\|]', a)[0].strip()
            if len(h) > 1 and len(a) > 1 and "PARTIDO" not in h.upper():
                matches.append((h, a))
                if len(matches) == 21: break # Explicitly target 21
        return matches
    except: return []

def resolve_matches(scraped, api_data):
    resolved = []
    api_map = {}
    for f in api_data:
        h, a = f['teams']['home']['name'], f['teams']['away']['name']
        fid = f['fixture']['id']
        api_map[f"{clean_name(h)} vs {clean_name(a)}"] = fid
        api_map[f"{clean_name(a)} vs {clean_name(h)}"] = fid
        
    api_keys = list(api_map.keys())
    print("\n🔍 RESOLVING 21-MATCH SLATE...")
    
    for i, (h, a) in enumerate(scraped):
        query = f"{clean_name(h)} vs {clean_name(a)}"
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_sort_ratio)
        
        if score > 60: # High tolerance for difficult matches
            fid = api_map[match]
            resolved.append(fid)
            print(f"Match {i+1:2}: '{h} vs {a}' -> RESOLVED (ID: {fid})")
        else:
            print(f"Match {i+1:2}: '{h} vs {a}' -> FAILED (Best Score: {score})")
    return resolved

if __name__ == "__main__":
    slate = scrape_full_slate()
    if not slate: print("Scrape failed."); exit(1)
    
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(slate, api_data)
    
    with open(CACHE_FILE, 'w') as f:
        json.dump({"last_updated": datetime.now().strftime("%Y-%m-%d"), "match_ids": ids}, f)
    print(f"\n🚀 Successfully resolved {len(ids)}/21 matches.")
