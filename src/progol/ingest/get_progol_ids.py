import requests
import re
import json
import os
import sys
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from thefuzz import process, fuzz
from dotenv import load_dotenv

from src.progol import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
PROGOL_CAT_URL = "https://quinielaposible.com/category/progol/"
CACHE_FILE = config.PROGOL_IDS_PATH
EXPECTED_SLATE_SIZE = 21

NICKNAME_MAP = {
    "ÁGUILAS": "AMÉRICA", "AGUILAS": "AMERICA", "C. AZUL": "CRUZ AZUL",
    "PUMAS": "UNAM PUMAS", "CHIVAS": "GUADALAJARA",
    "ÁLAS": "ATLAS", "TIGRES": "TIGRES UANL", "RAYO VALLEC": "RAYO VALLECANO",
    "ATH. BILBAO": "ATHLETIC CLUB", "PARÍS F.C.": "PARIS FC", "LOS ÁNGELES": "LOS ANGELES FC",
    "ST. LOUIS": "ST. LOUIS CITY", "NIZA": "NICE",
    "COLONIA": "KOLN", "PSV": "PSV EINDHOVEN", "OFI CRETA": "OFI",
    "ARIS": "ARIS THESSALONIKIS", "NAPOLES": "NAPOLI", "AUGSBURGO": "AUGSBURG",
    "W. BREMEN": "WERDER BREMEN", "B. MUNICH": "BAYERN MUNICH", "INTER P.A.": "INTERNACIONAL",
    "VASCO DA GA": "VASCO DA GAMA", "SP. LISBOA": "SPORTING CP", "S. LAGUNA": "SANTOS LAGUNA",
    "MAN. UNITED": "MANCHESTER UNITED",
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
        candidates = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            m = re.search(r'/progol-(\d+)[\w\-]*/?$', href)
            if m and not any(ext in href for ext in ['.png', '.jpg', '.jpeg', '.svg']):
                candidates.append((int(m.group(1)), href))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
    except: pass
    return PROGOL_CAT_URL

def scrape_flexible_slate(url):
    logging.info(f"Scraping official slate from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    matches = []
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Slate table on quinielaposible.com has 5 cols: [prob_L, LOCAL, prob_E, VISIT, prob_V]
        # 14 main rows + "Revancha" divider + 7 revancha rows = 21 unique matches.
        for table in soup.find_all('table'):
            local_matches = []
            for row in table.find_all('tr'):
                cols = [c.get_text(strip=True) for c in row.find_all('td')]
                if len(cols) == 5:
                    home, away = cols[1], cols[3]
                    if home and away and not home.replace('.', '').isdigit():
                        local_matches.append((home, away))
            if len(local_matches) >= 14:
                matches = local_matches
                break

        return matches[:21]
    except Exception as exc:
        logging.error(f"scrape_failed: {exc}")
        return []


def get_upcoming_api_fixtures(days_back=2, days_forward=5):
    """Fetch fixtures in a date window (played + unplayed). The previous version
    used `next=50` which silently dropped fixtures that had already kicked off,
    leaving the slate resolver with <21 matches by Saturday afternoon."""
    headers = {"x-apisports-key": API_KEY}
    today = datetime.now().date()
    date_from = (today - timedelta(days=days_back)).isoformat()
    date_to = (today + timedelta(days=days_forward)).isoformat()
    # Liga MX, Premier, La Liga, Serie A, Bundesliga, Ligue 1, MLS, Brasil, Argentina,
    # Portugal, Eredivisie, Belgium, Championship, La Liga 2, Libertadores, UCL, UEL,
    # Greek Super League, Bundesliga 2, Liga MX Expansion
    leagues = [262, 39, 140, 135, 78, 61, 253, 71, 128, 94, 88, 144, 40, 141, 13, 2, 3, 197, 79, 263]
    all_f = []
    for lid in leagues:
        for sn in [2025, 2026]:
            try:
                url = f"{BASE_URL}/fixtures?league={lid}&season={sn}&from={date_from}&to={date_to}"
                res = requests.get(url, headers=headers, timeout=20).json()
                all_f.extend(res.get('response', []))
            except Exception as exc:
                logging.warning(f"fixture_fetch_failed league={lid} season={sn}: {exc}")
                continue
    logging.info(f"Fetched {len(all_f)} candidate fixtures in window {date_from} -> {date_to}")
    return all_f

def resolve_matches(scraped, api_data, threshold=75):
    """Resolve scraped (home, away) pairs to API fixture IDs.

    Prefers matches that preserve home/away orientation. Returns flat list of fixture IDs
    (one per resolved match). Inverted matches are flagged in stdout so the user can
    sanity-check before predict runs.
    """
    fwd_map = {f"{clean_name(f['teams']['home']['name'])} vs {clean_name(f['teams']['away']['name'])}": f['fixture']['id'] for f in api_data}
    rev_map = {f"{clean_name(f['teams']['away']['name'])} vs {clean_name(f['teams']['home']['name'])}": f['fixture']['id'] for f in api_data}

    fwd_keys = list(fwd_map.keys())
    rev_keys = list(rev_map.keys())
    resolved = []
    print(f"\nRESOLVING 21-MATCH SLATE (threshold={threshold})...")
    for i, (h, v) in enumerate(scraped):
        query = f"{clean_name(h)} vs {clean_name(v)}"
        fwd_match, fwd_score = process.extractOne(query, fwd_keys, scorer=fuzz.token_sort_ratio) if fwd_keys else (None, 0)
        rev_match, rev_score = process.extractOne(query, rev_keys, scorer=fuzz.token_sort_ratio) if rev_keys else (None, 0)

        if fwd_score >= threshold and fwd_score >= rev_score:
            fid, inverted, score = fwd_map[fwd_match], False, fwd_score
        elif rev_score >= threshold:
            fid, inverted, score = rev_map[rev_match], True, rev_score
        else:
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> FAILED (best score: {max(fwd_score, rev_score)})")
            continue

        flag = "  [INVERTED - L/V swapped]" if inverted else ""
        print(f"Game {i+1:2}: {h:15} vs {v:15} -> ID {fid} (score {score}){flag}")
        resolved.append(fid)
    return resolved

if __name__ == "__main__":
    post_url = get_latest_progol_post_url()
    slate = scrape_flexible_slate(post_url)
    if not slate:
        logging.error("Slate not found.")
        sys.exit(1)

    api_data = get_upcoming_api_fixtures()
    ids = resolve_matches(slate, api_data)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "source_url": post_url,
        "expected": EXPECTED_SLATE_SIZE,
        "resolved": len(ids),
        "match_ids": ids,
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(payload, f, indent=2)

    logging.info(f"Resolved {len(ids)}/{EXPECTED_SLATE_SIZE} matches -> {CACHE_FILE}")
    if len(ids) != EXPECTED_SLATE_SIZE:
        logging.error(
            f"Slate incomplete: {len(ids)}/{EXPECTED_SLATE_SIZE} resolved. "
            f"Check NICKNAME_MAP and the date window in get_upcoming_api_fixtures."
        )
        sys.exit(2)
