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
        "Championship": 40, "Eredivisie": 88, "Liga MX Expansion": 263,
        "Russian Premier League": 235,
        # Domestic leagues that appear in Progol slates but were previously
        # unmodelled — matches in these leagues fell back to drift=True at
        # inference.
        "Belgium Jupiler": 144, "La Liga 2": 141, "Greek Super League": 197,
        "Bundesliga 2": 79, "Scottish Premiership": 179,
        "Chilean Primera": 265, "J1 League": 98, "J2 League": 99,
        # Cup competitions. Each match is flagged via the `is_cup` feature
        # (config.CUP_LEAGUE_IDS) so the model can learn cup-specific
        # patterns (knock-out pressure, mixed-tier opponents, lineup
        # rotation) while still sharing league-side history per club.
        "UEFA Champions League": 2, "UEFA Europa League": 3,
        "Copa Libertadores": 13, "Copa Sudamericana": 11,
        "FA Cup": 45, "EFL Cup": 48,
        "Copa del Rey": 143, "DFB-Pokal": 81,
        "Coppa Italia": 137, "Coupe de France": 66,
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
    """Fetches venue surface AND persists team metadata. Cached per team_id."""
    if team_id in venue_cache: return venue_cache[team_id]
    try:
        res = SESSION.get(f"{BASE_URL}/teams?id={team_id}", timeout=30).json().get('response', [])
        if res:
            payload = res[0]
            database.upsert_team(team_id, payload)
            v_id = payload['venue']['id']
            v_surf = payload['venue']['surface']
            venue_cache[team_id] = (v_id, v_surf)
            return (v_id, v_surf)
    except: pass
    return (0, "Unknown")

def _safe_float(v):
    """API-Football returns numeric stats as either strings or numbers; the
    expected_goals field in particular comes as 'None'/None/'1.20'/1.2.
    Returns float or None — None means caller should fall back."""
    if v is None or v == 'None':
        return None
    try:
        return float(str(v).replace('%', ''))
    except (TypeError, ValueError):
        return None


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
        stats['xg_real'] = False  # default — flipped below if API provides actual xG
        if s_res:
            for i, ts in enumerate(s_res):
                p = 'h' if i == 0 else 'a'
                s_map = {item['type']: item['value'] for item in ts['statistics']}
                stats[f'{p}_sh'] = int(s_map.get('Shots on Goal', 0) or 0)
                stats[f'{p}_po'] = int(str(s_map.get('Ball Possession', "0") or "0").replace('%',''))
                stats[f'{p}_co'] = int(s_map.get('Corner Kicks', 0) or 0)
                total_sh = int(s_map.get('Total Shots', 0) or 0)
                # API-Football publishes real `expected_goals` for top-5
                # leagues from 2017+. Prefer it; fall back to the legacy
                # proxy `0.3*shots_on_goal + 0.1*total_shots` so the column
                # is never NULL. xg_real flag lets downstream code weight
                # rows by data quality if useful.
                real_xg = _safe_float(s_map.get('expected_goals'))
                if real_xg is not None:
                    stats[f'{p}_xg'] = real_xg
                    stats['xg_real'] = True
                else:
                    stats[f'{p}_xg'] = (stats[f'{p}_sh'] * 0.3) + (total_sh * 0.1)

        o_res = SESSION.get(f"{BASE_URL}/odds?fixture={fid}&bookmaker=8", timeout=30).json().get('response', [])
        if o_res and o_res[0].get('bookmakers'):
            bets = o_res[0]['bookmakers'][0]['bets'][0]['values']
            stats['o_h'], stats['o_d'], stats['o_a'] = float(bets[0]['odd']), float(bets[1]['odd']), float(bets[2]['odd'])
        else:
            stats['o_h'], stats['o_d'], stats['o_a'] = 0.0, 0.0, 0.0

        # Injuries — players unavailable at fixture time. API-Football's
        # /injuries endpoint returns each item with team.id, so we split
        # by home/away. For old fixtures the API often returns empty —
        # treated as 0 injuries (best-effort: no data ≠ no injuries, but
        # we can't distinguish, and 0 is the safer default than NULL).
        stats['h_inj'], stats['a_inj'] = 0, 0
        try:
            inj_res = SESSION.get(f"{BASE_URL}/injuries?fixture={fid}", timeout=30).json().get('response', [])
            for item in inj_res:
                tid = (item.get('team') or {}).get('id')
                if tid == h_id:
                    stats['h_inj'] += 1
                elif tid == a_id:
                    stats['a_inj'] += 1
        except Exception:
            pass

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

def backfill_teams(max_workers=10):
    """One-time fetch of team metadata for any team_id seen in matches but
    not yet present in the teams table. Idempotent — safe to call every run."""
    conn = database.get_connection()
    rows = conn.execute('''
        SELECT DISTINCT team_id FROM (
            SELECT home_id AS team_id FROM matches
            UNION SELECT away_id AS team_id FROM matches
        )
        WHERE team_id IS NOT NULL
          AND team_id NOT IN (SELECT team_id FROM teams)
    ''').fetchall()
    conn.close()
    team_ids = [r[0] for r in rows]
    if not team_ids:
        return
    logging.info(f"backfill_teams: {len(team_ids)} teams to fetch")

    def fetch_one(tid):
        try:
            res = SESSION.get(f"{BASE_URL}/teams?id={tid}", timeout=30).json().get('response', [])
            if res:
                database.upsert_team(tid, res[0])
                return tid, True
        except Exception:
            pass
        return tid, False

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed({ex.submit(fetch_one, tid): tid for tid in team_ids}):
            _, ok = fut.result()
            done += 1
            if done % 100 == 0:
                logging.info(f"backfill_teams: {done}/{len(team_ids)}")


def enrich_database_alpha(max_workers=10):
    # Backfill scope: any FT match without a Batch-A timestamp. This covers
    # (a) freshly discovered matches (no enrichment of any kind) and
    # (b) historically-enriched rows that pre-date xG-real/injuries and
    # therefore need a re-pull. update_alpha_stats sets alpha_v2_at, so
    # the loop converges on a per-row "done" marker rather than the old
    # heuristic of "odds_home IS NULL" (which misses pre-existing rows).
    while True:
        conn = database.get_connection()
        query = """
            SELECT fixture_id FROM matches
            WHERE status = 'FT' AND alpha_v2_at IS NULL
            LIMIT 100
        """
        fixtures = pd.read_sql_query(query, conn)['fixture_id'].tolist()
        conn.close()
        if not fixtures: break
        logging.info(f"alpha_enrichment_batch", extra={'size': len(fixtures)})
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fid = {executor.submit(fetch_alpha_details, fid): fid for fid in fixtures}
            for future in as_completed(future_to_fid):
                fid, data = future.result()
                if data:
                    database.update_alpha_stats(fid, data)
                else:
                    # Mark this fixture as attempted so the loop converges,
                    # but DO NOT overwrite existing fields. A prior enrichment
                    # may have left valid odds/xg in place; a transient API
                    # miss shouldn't wipe them.
                    database.mark_alpha_v2_tried(fid)
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

    # Step 3: Backfill team metadata (names, country, venue) for any team_id
    # not yet in the teams table. Cheap when up-to-date.
    backfill_teams(max_workers=10)
