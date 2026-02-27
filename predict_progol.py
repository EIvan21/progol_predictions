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

def fetch_hyper_context(match_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        lid = match['league']['id']
        
        def get_team_stats(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0.3
            stats = []
            for g in reversed(data):
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                stats.append({'gf':gf, 'ga':ga, 'win': 1 if gf>ga else 0, 'cs': 1 if ga==0 else 0})
            df = pd.DataFrame(stats)
            return df['gf'].mean(), df['ga'].mean(), df['cs'].mean(), df['win'].mean()

        h_gf, h_ga, h_cs, h_p = get_team_stats(h_id)
        a_gf, a_ga, a_cs, a_p = get_team_stats(a_id)
        
        return {
            'league_id': lid, 'league_ha_factor': 0.45, # Global average home win rate
            'venue_encoded': 0.45, 'ref_encoded': 0.33, # Smooth defaults
            'roll_gf_home': h_gf, 'roll_ga_home': h_ga, 'cs_rate_home': h_cs, 'power_score_home': h_p,
            'roll_gf_away': a_gf, 'roll_ga_away': a_ga, 'cs_rate_away': a_cs, 'power_score_away': a_p
        }
    except Exception as e: return None

def predict_progol(match_ids):
    if not os.path.exists(METRICS_PATH): return
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics['features']
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_hyper_context(mid)
        if data:
            df_in = pd.DataFrame([data])
            for col in FEATURES:
                if col not in df_in.columns: df_in[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(df_in[FEATURES]), columns=FEATURES)
            probs = model.predict_proba(X_scaled)[0]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}%")

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
        predict_progol(ids)
