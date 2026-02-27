import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle

load_dotenv()
MODEL_PATH = 'models/progol_model.bin'
ENSEMBLE_PATH = 'models/ensemble_models.pkl'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'models/metrics.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_match_context(match_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        
        def get_team_stats(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0
            stats = []
            for g in reversed(data):
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                stats.append({'gf':gf, 'ga':ga, 'cs': 1 if ga==0 else 0})
            df = pd.DataFrame(stats).rolling(5, min_periods=1).mean().iloc[-1]
            return df['gf'], df['ga'], df['cs']

        h_stats = get_team_stats(h_id)
        a_stats = get_team_stats(a_id)
        
        return {
            'league_id': match['league']['id'],
            'season': match['league']['season'],
            'venue': hash(match['fixture']['venue']['name'] or "Unknown") % 1000,
            'referee': hash(match['fixture']['referee'] or "Unknown") % 1000,
            'roll_gf_home': h_stats[0], 'roll_ga_home': h_stats[1], 'clean_sheet_rate_home': h_stats[2],
            'roll_gf_away': a_stats[0], 'roll_ga_away': a_stats[1], 'clean_sheet_rate_away': a_stats[2],
            'days_rest_home': 7, 'days_rest_away': 7, 'h2h_home_win_rate': 0.33
        }
    except Exception as e:
        print(f"Error fetching match data: {e}"); return None

def predict_progol(match_ids):
    if not os.path.exists(METRICS_PATH): return
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics['features']
    
    if not os.path.exists(SCALER_PATH): return
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

    is_ensemble = os.path.exists(ENSEMBLE_PATH)
    if is_ensemble:
        with open(ENSEMBLE_PATH, 'rb') as f: models = pickle.load(f)
    else:
        with open(MODEL_PATH, 'rb') as f: models = {"Single": pickle.load(f)}

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_match_context(mid)
        if data:
            # Ensure all features exist in the fetched data
            df_input = pd.DataFrame([data])
            for col in FEATURES:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            X = df_input[FEATURES]
            X_scaled = scaler.transform(X)
            
            all_probs = []
            for m in models.values():
                all_probs.append(m.predict_proba(X_scaled)[0])
            
            final_probs = np.mean(all_probs, axis=0)
            print(f"{mid:<10} | {final_probs[0]*100:8.2f}% | {final_probs[1]*100:8.2f}% | {final_probs[2]*100:8.2f}%")

if __name__ == "__main__":
    ids_file = 'current_progol_ids.json'
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f: ids = json.load(f)
        predict_progol(ids)
