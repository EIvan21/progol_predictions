import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
MODEL_PATH = 'models/progol_model.bin'
ENSEMBLE_PATH = 'models/ensemble_models.pkl'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'models/metrics.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_elite_context(match_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        
        def get_team_elite_stats(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0.3 # gf, ga, cs, power
            
            stats = []
            for g in reversed(data):
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                stats.append({'gf':gf, 'ga':ga, 'win': 1 if gf>ga else 0, 'cs': 1 if ga==0 else 0})
            
            df = pd.DataFrame(stats)
            power = df['win'].rolling(10, min_periods=1).mean().iloc[-1]
            # Simple adjusted goals for prediction (since we don't have all upcoming opponent stats easily)
            gf_roll = df['gf'].rolling(5, min_periods=1).mean().iloc[-1]
            ga_roll = df['ga'].rolling(5, min_periods=1).mean().iloc[-1]
            cs_roll = df['cs'].rolling(5, min_periods=1).mean().iloc[-1]
            return gf_roll, ga_roll, cs_roll, power

        h_gf, h_ga, h_cs, h_p = get_team_elite_stats(h_id)
        a_gf, a_ga, a_cs, a_p = get_team_elite_stats(a_id)
        
        return {
            'league_id': match['league']['id'],
            'venue': hash(match['fixture']['venue']['name'] or "Unknown") % 1000,
            'referee': hash(match['fixture']['referee'] or "Unknown") % 1000,
            'roll_adj_gf_home': h_gf * (1 + a_p),
            'roll_adj_ga_home': h_ga * (2 - a_p),
            'clean_sheet_rate_home': h_cs,
            'power_score_home': h_p,
            'days_rest_home': 7,
            'roll_adj_gf_away': a_gf * (1 + h_p),
            'roll_adj_ga_away': a_ga * (2 - h_p),
            'clean_sheet_rate_away': a_cs,
            'power_score_away': a_p,
            'days_rest_away': 7
        }
    except Exception as e:
        print(f"Error: {e}"); return None

def predict_progol(match_ids):
    if not os.path.exists(METRICS_PATH): return
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics['features']
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

    if os.path.exists(ENSEMBLE_PATH):
        with open(ENSEMBLE_PATH, 'rb') as f: models = pickle.load(f)
    else:
        with open(MODEL_PATH, 'rb') as f: models = {"Single": pickle.load(f)}

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_elite_context(mid)
        if data:
            df_in = pd.DataFrame([data])
            for col in FEATURES:
                if col not in df_in.columns: df_in[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(df_in[FEATURES]), columns=FEATURES)
            all_probs = [m.predict_proba(X_scaled)[0] for m in models.values()]
            final_probs = np.mean(all_probs, axis=0)
            print(f"{mid:<10} | {final_probs[0]*100:8.2f}% | {final_probs[1]*100:8.2f}% | {final_probs[2]*100:8.2f}%")

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
        predict_progol(ids)
