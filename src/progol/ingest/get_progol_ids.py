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

NICKNAME_MAP = {
    "ÁGUILAS": "AMÉRICA", "C. AZUL": "CRUZ AZUL", "PUMAS": "UNAM PUMAS", "CHIVAS": "GUADALAJARA",
    "ÁLAS": "ATLAS", "TIGRES": "TIGRES UANL", "RAYO VALLEC": "RAYO VALLECANO", "ATH. BILBAO": "ATHLETIC CLUB",
    "PARÍS F.C.": "PARIS FC", "LOS ÁNGELES": "LOS ANGELES FC", "ST. LOUIS": "ST. LOUIS CITY", "NIZA": "NICE"
}

def clean_name(name):
    name = name.upper().strip()
    for nickname, official in NICKNAME_MAP.items():
        if nickname in name: return official
    removals = ["FC", "CF", "CD", "UD", "CLUB", "DEPORTIVO", "REAL", "ATLETICO", "S.C."]
    for r in removals: name = name.replace(r, "").strip()
    return name

def get_latest_progol_post_url():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(PROGOL_CAT_URL, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link['href']
            # Search for patterns like pronostico-progol-2322/ or progol-2322/
            if re.search(r'progol-\d+/?$', href) and not any(ext in href for ext in ['.png', '.jpg', '.jpeg', '.svg']):
                return href
    except: pass
    return PROGOL_CAT_URL

def scrape_flexible_slate(url):
    logging.info(f"Scraping official slate from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    matches = []
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. Try Table Strategy first
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    t1, t2 = cols[0].get_text(strip=True), cols[-1].get_text(strip=True)
                    if len(t1) > 2 and len(t2) > 2 and "VS" not in t1.upper() and "PARTIDO" not in t1.upper():
                        t1, t2 = re.split(r'(\d|:)', t1)[0].strip(), re.split(r'(\d|:)', t2)[0].strip()
                        matches.append((t1, t2))
            if len(matches) >= 14: break
            
        # 2. Fallback to Paragraph Regex Strategy if table was empty
        if not matches:
            content = soup.get_text("\n")
            # Look for "TEAM A vs TEAM B"
            patterns = re.findall(r'([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)\s+vs\s+([A-ZÁÉÍÓÚÑ0-9\s\.\-]+)', content)
            for h, a in patterns:
                h = h.strip().split('\n')[-1].strip()
                # Remove date/time from away team (split by dash or numbers)
                a = re.split(r'[\u2013\-\|0-9]', a)[0].strip()
                if len(h) > 1 and len(a) > 1 and "PARTIDO" not in h.upper():
                    matches.append((h, a))
                    if len(matches) == 21: break
                    
        return matches[:21]
    except: return []

def get_upcoming_api_fixtures(days=14):
    headers = {"x-apisports-key": API_KEY}
    start = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    leagues = [262, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 40, 141, 13, 2, 3]
    all_f = []
    for lid in leagues:
        for sn in [2025, 2026]:
            try:
                res = requests.get(f"{BASE_URL}/fixtures?league={lid}&season={sn}&next=50", headers=headers).json()
                if res.get('response'): all_f.extend(res['response'])
            except: continue
    return all_f

def resolve_matches(scraped, api_data):
    resolved = []
    api_map = {f"{clean_name(f['teams']['home']['name'])} vs {clean_name(f['teams']['away']['name'])}": f['fixture']['id'] for f in api_data}
    api_map.update({f"{clean_name(f['teams']['away']['name'])} vs {clean_name(f['teams']['home']['name'])}": f['fixture']['id'] for f in api_data})
    
    api_keys = list(api_map.keys())
    print("\n🔍 RESOLVING 21-MATCH SLATE...")
    for i, (h, v) in enumerate(scraped):
        query = f"{clean_name(h)} vs {clean_name(v)}"
        match, score = process.extractOne(query, api_keys, scorer=fuzz.token_sort_ratio)
        if score > 60:
            fid = api_map[match]
            resolved.append(fid)
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> Resolved (ID: {fid})")
        else:
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> ❌ FAILED (Score: {score})")
    return resolved

if __name__ == "__main__":
    post_url = get_latest_progol_post_url()
    slate = scrape_flexible_slate(post_url)
    if not slate: print("❌ Slate not found."); exit(1)
    
    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(slate, api_data)
    with open(CACHE_FILE, 'w') as f: json.dump({"last_updated": datetime.now().strftime("%Y-%m-%d"), "match_ids": ids}, f)
    print(f"\n🚀 Successfully resolved {len(ids)} matches.")
