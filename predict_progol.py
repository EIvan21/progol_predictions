import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle
import warnings

warnings.filterwarnings("ignore")
load_dotenv()
MODEL_PATH = 'models/progol_stack_model.bin'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'models/metrics.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_match_stats(team_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        url = f"{BASE_URL}/fixtures?team={team_id}&last=5&status=FT"
        res = requests.get(url, headers=headers).json().get('response', [])
        if not res: return 0, 0, 0, 0, 0, 0
        
        match_ids = [m['fixture']['id'] for m in res]
        stats_list = []
        
        for fid in match_ids:
            s_url = f"{BASE_URL}/fixtures/statistics?fixture={fid}"
            s_res = requests.get(s_url, headers=headers).json().get('response', [])
            if not s_res: continue
            
            # Find stats for our team
            team_stats = [s for s in s_res if s['team']['id'] == team_id][0]
            opp_stats = [s for s in s_res if s['team']['id'] != team_id][0]
            
            s_map = {item['type']: item['value'] for item in team_stats['statistics']}
            o_map = {item['type']: item['value'] for item in opp_stats['statistics']}
            
            stats_list.append({
                'gf': int(s_map.get('Goals', 0) or 0),
                'ga': int(o_map.get('Goals', 0) or 0),
                'shots': int(s_map.get('Shots on Goal', 0) or 0),
                'poss': int(str(s_map.get('Ball Possession', "0") or "0").replace('%','')),
                'corners': int(s_map.get('Corner Kicks', 0) or 0)
            })
            time.sleep(0.5) # Rate limit safety

        df = pd.DataFrame(stats_list).mean()
        cs_rate = (pd.DataFrame(stats_list)['ga'] == 0).mean()
        return df['gf'], df['ga'], df['shots'], df['poss'], df['corners'], cs_rate
    except: return 0, 0, 0, 0, 0, 0

def predict_progol(match_ids):
    if not os.path.exists(METRICS_PATH): return
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics['features']
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    headers = {"x-apisports-key": API_KEY}
    for mid in match_ids:
        try:
            m_res = requests.get(f"{BASE_URL}/fixtures?id={mid}", headers=headers).json()['response'][0]
            h_id, a_id = m_res['teams']['home']['id'], m_res['teams']['away']['id']
            
            h_s = fetch_match_stats(h_id)
            a_s = fetch_match_stats(a_id)
            
            data = {
                'league_id': m_res['league']['id'],
                'venue_encoded': 0.45, 'ref_encoded': 0.33, 'league_ha_factor': 0.45,
                'roll_gf_home': h_s[0], 'roll_ga_home': h_s[1], 'roll_shots_home': h_s[2], 'roll_poss_home': h_s[3], 'roll_corners_home': h_s[4], 'cs_rate_home': h_s[5],
                'roll_gf_away': a_s[0], 'roll_ga_away': a_s[1], 'roll_shots_away': a_s[2], 'roll_poss_away': a_s[3], 'roll_corners_away': a_s[4], 'cs_rate_away': a_s[5]
            }
            
            X = pd.DataFrame([data])
            for col in FEATURES:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
            probs = model.predict_proba(X_scaled)[0]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}%")
        except: print(f"{mid:<10} | Error fetching live stats.")

if __name__ == "__main__":
    import time
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
        predict_progol(ids)
