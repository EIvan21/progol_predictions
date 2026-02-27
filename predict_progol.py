import os
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv

# Config
load_dotenv()
MODEL_PATH = 'models/progol_xgb_model.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

FEATURES = [
    'league_id', 'venue', 'referee',
    'avg_goals_scored_home', 'avg_goals_conceded_home', 'days_rest_home',
    'avg_goals_scored_away', 'avg_goals_conceded_away', 'days_rest_away'
]

def fetch_real_rolling_stats(match_id):
    """Fetches real rolling stats for a specific fixture from the API."""
    headers = {"x-apisports-key": API_KEY}
    
    # 1. Get Fixture Details
    res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
    if not res['response']: return None
    
    match = res['response'][0]
    h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
    
    def get_team_avg(team_id):
        # Get last 5 matches for this team
        url = f"{BASE_URL}/fixtures?team={team_id}&last=5&status=FT"
        games = requests.get(url, headers=headers).json()['response']
        
        goals_for, goals_against = [], []
        for g in games:
            is_home = g['teams']['home']['id'] == team_id
            goals_for.append(g['goals']['home'] if is_home else g['goals']['away'])
            goals_against.append(g['goals']['away'] if is_home else g['goals']['home'])
            
        return np.mean(goals_for), np.mean(goals_against)

    h_avg_f, h_avg_a = get_team_avg(h_id)
    a_avg_f, a_avg_a = get_team_avg(a_id)
    
    return {
        'league_id': match['league']['id'],
        'venue': hash(match['fixture']['venue']['name']) % 1000, # Consistent encoding
        'referee': hash(match['fixture']['referee']) % 1000,
        'avg_goals_scored_home': h_avg_f, 'avg_goals_conceded_home': h_avg_a, 'days_rest_home': 7,
        'avg_goals_scored_away': a_avg_f, 'avg_goals_conceded_away': a_avg_a, 'days_rest_away': 7
    }

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH): return
    model = xgb.XGBClassifier(); model.load_model(MODEL_PATH)
    
    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_real_rolling_stats(mid)
        if data:
            X = pd.DataFrame([data])[FEATURES]
            probs = model.predict_proba(X)[0]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}%")
        else:
            print(f"{mid:<10} | Match data not found.")

if __name__ == "__main__":
    import sys
    
    # 1. Try to load IDs from the automated scraper output
    ids_file = 'current_progol_ids.json'
    
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            PROGOL_MATCH_IDS = json.load(f)
        print(f"Loaded {len(PROGOL_MATCH_IDS)} match IDs from {ids_file}")
    else:
        # Fallback: Manual IDs if file is missing
        print(f"Warning: {ids_file} not found. Using default placeholder IDs.")
        PROGOL_MATCH_IDS = [1228220, 1208154, 1208155, 1208156, 1208157]

    if PROGOL_MATCH_IDS:
        predict_progol(PROGOL_MATCH_IDS)
    else:
        print("No match IDs available to predict.")
