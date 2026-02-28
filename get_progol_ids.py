import requests
import re
import json
import os
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from thefuzz import process, fuzz
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
PROGOL_CAT_URL = "https://quinielaposible.com/category/progol/"
CACHE_FILE = 'current_progol_ids.json'

# Hardcoded Dictionary for Progol Nicknames
NICKNAME_MAP = {
    "ÁGUILAS": "AMÉRICA",
    "C. AZUL": "CRUZ AZUL",
    "PUMAS": "UNAM PUMAS",
    "CHIVAS": "GUADALAJARA",
    "ÁLAS": "ATLAS",
    "TIGRES": "TIGRES UANL",
    "RAYO VALLEC": "RAYO VALLECANO",
    "ATH. BILBAO": "ATHLETIC CLUB",
    "PARÍS F.C.": "PARIS FC",
    "LOS ÁNGELES": "LOS ANGELES FC",
    "ST. LOUIS": "ST. LOUIS CITY",
    "NIZA": "NICE"
}

def clean_name(name):
    name = name.upper().strip()
    # Replace common abbreviations
    for nickname, official in NICKNAME_MAP.items():
        if nickname in name:
            return official
    # Remove junk
    removals = ["FC", "CF", "CD", "UD", "CLUB", "DEPORTIVO", "REAL", "ATLETICO", "S.C."]
    for r in removals:
        name = name.replace(r, "").strip()
    return name

def get_latest_progol_post_url():
    """Finds the URL of the most recent specific Progol post."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(PROGOL_CAT_URL, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Find the first article link that contains 'progol-' and numbers
        links = soup.find_all('a', href=re.compile(r'/progol-\d+'))
        if links:
            return links[0]['href']
    except: pass
    return PROGOL_CAT_URL # Fallback

def scrape_official_table(url):
    """Parses the structured HTML table from the specific post."""
    logging.info(f"Scraping official table from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    matches = []
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                # Progol tables usually have 3+ columns (Home, 'vs', Away)
                if len(cols) >= 3:
                    t1 = cols[0].get_text(strip=True)
                    t2 = cols[-1].get_text(strip=True)
                    
                    if len(t1) > 2 and len(t2) > 2 and "VS" not in t1.upper():
                        # Basic cleaning of date/time if it's attached
                        t1 = re.split(r'\d', t1)[0].strip()
                        t2 = re.split(r'\d', t2)[0].strip()
                        matches.append((t1, t2))
            if len(matches) >= 14: break # Found the main quiniela
            
        return matches[:21] # Return up to 21
    except: return []

def get_upcoming_api_fixtures(days=14):
    headers = {"x-apisports-key": API_KEY}
    start = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    all_fixtures = []
    
    # Check 2025 and 2026 seasons for all major leagues
    leagues = [262, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 40, 141]
    for lid in leagues:
        for sn in [2025, 2026]:
            try:
                url = f"{BASE_URL}/fixtures?league={lid}&season={sn}&next=50"
                res = requests.get(url, headers=headers).json()
                if res.get('response'): all_fixtures.extend(res['response'])
            except: continue
    return all_fixtures

def resolve_matches(scraped, api_data):
    resolved = []
    api_map = {f"{clean_name(f['teams']['home']['name'])} vs {clean_name(f['teams']['away']['name'])}": f['fixture']['id'] for f in api_data}
    api_keys = list(api_map.keys())
    
    print("\n" + "="*20 + " RESOLVING 21-MATCH SLATE " + "="*20)
    for i, (h, v) in enumerate(scraped):
        query = f"{clean_name(h)} vs {clean_name(v)}"
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_sort_ratio)
        
        if score > 60:
            fid = api_map[match]
            resolved.append(fid)
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> Found ID: {fid}")
        else:
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> ❌ FAILED (Best Score: {score})")
    return resolved

if __name__ == "__main__":
    post_url = get_latest_progol_post_url()
    slate = scrape_official_table(post_url)
    
    if not slate:
        print("❌ Could not find official table. Check source URL.")
        exit(1)
        
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(slate, api_data)
    
    with open(CACHE_FILE, 'w') as f:
        json.dump({"last_updated": datetime.now().strftime("%Y-%m-%d"), "match_ids": ids}, f)
    print(f"\n🚀 Successfully resolved {len(ids)} total matches.")
