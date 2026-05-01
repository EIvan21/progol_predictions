import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.progol import config, database
from src.progol.utils.http import api_football_session
from src.progol.utils.logging_setup import configure as configure_logging
from concurrent.futures import ThreadPoolExecutor, as_completed

configure_logging()
load_dotenv()
SESSION = api_football_session()
BASE_URL = "https://v3.football.api-sports.io"

if config.IS_LOCAL_TEST:
    LEAGUES = {"Liga MX": 262}
    SEASONS = [2025]
    logging.info("IS_LOCAL_TEST=True -> fetching only Liga MX 2025")
else:
    LEAGUES = {
        "Liga MX": 262, "Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78,
        "Ligue 1": 61, "MLS": 253, "Brazil Serie A": 71, "Argentina": 128, "Portugal": 94,
        "Championship": 40, "Eredivisie": 88, "Liga MX Expansion": 263
    }
    SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

# GLOBAL CACHES
standings_cache = {}
venue_cache = {}

def get_standings(league, season):
    """Fetches and caches standings for a given league/season."""
    key = f"{league}_{season}"
    if key in standings_cache: return standings_cache[key]
    
    try:
        res = SESSION.get(f"{BASE_URL}/standings?league={league}&season={season}", timeout=30).json().get('response', [])
        if res:
            table = {}
            # Some leagues might have multiple groups (e.g. Apertura/Clausura), take the first one
            for standing in res[0]['league']['standings'][0]:
                table[standing['team']['id']] = {
                    'rank': standing['rank'],
                    'form': standing['form']
                }
            standings_cache[key] = table
            return table
    except: pass
    return {}

def get_h2h(tid1, tid2):
    """Fetches head-to-head stats (wins, draws, losses)."""
    try:
        res = SESSION.get(f"{BASE_URL}/fixtures/headtohead?h2h={tid1}-{tid2}", timeout=30).json().get('response', [])
        h, d, a = 0, 0, 0
        for m in res[:10]: # Last 10 matches
            if m['goals']['home'] > m['goals']['away']: h += 1
            elif m['goals']['home'] == m['goals']['away']: d += 1
            else: a += 1
        return h, d, a
    except: return 0, 0, 0

def get_venue_surface(team_id):
    """Fetches and caches venue surface (grass vs artificial)."""
    if team_id in venue_cache: return venue_cache[team_id]
    try:
        res = SESSION.get(f"{BASE_URL}/teams?id={team_id}", timeout=30).json().get('response', [])
        if res:
            v_id = res[0]['venue']['id']
            v_surf = res[0]['venue']['surface']
            venue_cache[team_id] = (v_id, v_surf)
            return (v_id, v_surf)
    except: pass
    return (0, "Unknown")

def fetch_alpha_details(fid):
    try:
        # 1. Get Match Teams & IDs first
        conn = database.get_connection()
        m_info = pd.read_sql_query(f"SELECT home_id, away_id, league_id, season FROM matches WHERE fixture_id = {fid}", conn).iloc[0]
        conn.close()
        h_id, a_id, lid, season = int(m_info['home_id']), int(m_info['away_id']), int(m_info['league_id']), int(m_info['season'])

        # 2. Statistics & Odds
        s_res = SESSION.get(f"{BASE_URL}/fixtures/statistics?fixture={fid}", timeout=30).json().get('response', [])
        stats = {}
        if s_res:
            for i, ts in enumerate(s_res):
                p = 'h' if i == 0 else 'a'
                s_map = {item['type']: item['value'] for item in ts['statistics']}
                stats[f'{p}_sh'] = int(s_map.get('Shots on Goal', 0) or 0)
                stats[f'{p}_po'] = int(str(s_map.get('Ball Possession', "0") or "0").replace('%',''))
                stats[f'{p}_co'] = int(s_map.get('Corner Kicks', 0) or 0)
                total_sh = int(s_map.get('Total Shots', 0) or 0)
                stats[f'{p}_xg'] = (stats[f'{p}_sh'] * 0.3) + (total_sh * 0.1)

        o_res = SESSION.get(f"{BASE_URL}/odds?fixture={fid}&bookmaker=8", timeout=30).json().get('response', [])
        if o_res and o_res[0].get('bookmakers'):
            bets = o_res[0]['bookmakers'][0]['bets'][0]['values']
            stats['o_h'], stats['o_d'], stats['o_a'] = float(bets[0]['odd']), float(bets[1]['odd']), float(bets[2]['odd'])
        else:
            stats['o_h'], stats['o_d'], stats['o_a'] = 0.0, 0.0, 0.0

        # 3. New Strategic Context (Rankings, Form, H2H, Venue)
        std = get_standings(lid, season)
        stats['h_rank'] = std.get(h_id, {}).get('rank', 10)
        stats['a_rank'] = std.get(a_id, {}).get('rank', 10)
        stats['h_form'] = std.get(h_id, {}).get('form', "DDDDD")
        stats['a_form'] = std.get(a_id, {}).get('form', "DDDDD")
        
        v_id, v_surf = get_venue_surface(h_id)
        stats['v_id'], stats['v_surf'] = v_id, v_surf
        
        h2h_h, h2h_d, h2h_a = get_h2h(h_id, a_id)
        stats['h2h_h'], stats['h2h_d'], stats['h2h_a'] = h2h_h, h2h_d, h2h_a

        return fid, stats
    except: return fid, None

def enrich_database_alpha(max_workers=10):
    while True:
        conn = database.get_connection()
        query = "SELECT fixture_id FROM matches WHERE status = 'FT' AND odds_home IS NULL LIMIT 100"
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        if not fixtures: break
        logging.info(f"alpha_enrichment_batch", extra={'size': len(fixtures)})
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fid = {executor.submit(fetch_alpha_details, fid): fid for fid in fixtures}
            for future in as_completed(future_to_fid):
                fid, data = future.result()
                if data: database.update_alpha_stats(fid, data)
                else: database.update_alpha_stats(fid, {'o_h': 0})
        time.sleep(0.5)

if __name__ == "__main__":
    database.init_db()
    # RESTORED: Fixture Discovery Logic
    logging.info("Step 1: Discovering Matches (2019-2026)...")
    for name, lid in LEAGUES.items():
        for season in SEASONS:
            last_date = database.get_latest_match_date(lid, season)
            params = {"league": lid, "season": season}
            if last_date:
                start = (datetime.strptime(last_date[:10], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                params["from"], params["to"] = start, datetime.now().strftime("%Y-%m-%d")
                if start > params["to"]: continue
            
            logging.info(f"Fetching {name} {season} fixtures...")
            try:
                res = SESSION.get(f"{BASE_URL}/fixtures", params=params, timeout=30).json()
                matches = res.get('response', [])
                if matches:
                    database.save_matches_to_db(matches, season)
                    logging.info(f"✅ Added {len(matches)} matches.")
                time.sleep(1.2)
            except: continue

    # Step 2: Turbo Alpha Enrichment
    enrich_database_alpha(max_workers=50)
