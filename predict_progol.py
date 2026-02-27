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
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

FEATURES = [
    'league_id', 'venue', 'referee',
    'ewm_goals_for_home', 'ewm_goals_against_home', 'ewm_goal_diff_home', 'ewm_form_points_home', 'days_rest_home',
    'ewm_goals_for_away', 'ewm_goals_against_away', 'ewm_goal_diff_away', 'ewm_form_points_away', 'days_rest_away'
]

def fetch_real_rolling_stats(match_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        
        def get_team_stats(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0
            stats = []
            for g in reversed(data):
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                gd = gf - ga
                stats.append({'gf':gf, 'ga':ga, 'gd':gd, 'pts': (3 if gd>0 else (1 if gd==0 else 0))})
            ewm = pd.DataFrame(stats).ewm(span=5).mean().iloc[-1]
            return ewm['gf'], ewm['ga'], ewm['gd'], ewm['pts']

        h_stats = get_team_stats(h_id)
        a_stats = get_team_stats(a_id)
        
        return {
            'league_id': match['league']['id'],
            'venue': hash(match['fixture']['venue']['name'] or "Unknown") % 1000,
            'referee': hash(match['fixture']['referee'] or "Unknown") % 1000,
            'ewm_goals_for_home': h_stats[0], 'ewm_goals_against_home': h_stats[1], 'ewm_goal_diff_home': h_stats[2], 'ewm_form_points_home': h_stats[3], 'days_rest_home': 7,
            'ewm_goals_for_away': a_stats[0], 'ewm_goals_against_away': a_stats[1], 'ewm_goal_diff_away': a_stats[2], 'ewm_form_points_away': a_stats[3], 'days_rest_away': 7
        }
    except Exception as e:
        print(f"Error: {e}"); return None

def predict_progol(match_ids):
    # Load Scaler
    if not os.path.exists(SCALER_PATH): return
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

    # Load Model(s)
    is_ensemble = os.path.exists(ENSEMBLE_PATH)
    if is_ensemble:
        print("Using ENSEMBLE mode (Collective Voting)...")
        with open(ENSEMBLE_PATH, 'rb') as f: models = pickle.load(f)
    elif os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f: models = {"Single": pickle.load(f)}
    else:
        print("No models found."); return

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_real_rolling_stats(mid)
        if data:
            X = scaler.transform(pd.DataFrame([data])[FEATURES])
            
            # Combine probabilities from all models
            all_probs = []
            for m in models.values():
                all_probs.append(m.predict_proba(X)[0])
            
            final_probs = np.mean(all_probs, axis=0)
            print(f"{mid:<10} | {final_probs[0]*100:8.2f}% | {final_probs[1]*100:8.2f}% | {final_probs[2]*100:8.2f}%")

if __name__ == "__main__":
    ids_file = 'current_progol_ids.json'
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f: ids = json.load(f)
        predict_progol(ids)
