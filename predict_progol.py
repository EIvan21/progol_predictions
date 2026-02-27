import os
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
import logging

# Config
load_dotenv()
MODEL_PATH = 'models/progol_xgb_model.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

# Features must match train_model.py exactly
FEATURES = [
    'league_id', 'venue', 'referee',
    'ewm_goals_for_home', 'ewm_goals_against_home', 'ewm_goal_diff_home', 'ewm_form_points_home', 'days_rest_home',
    'ewm_goals_for_away', 'ewm_goals_against_away', 'ewm_goal_diff_away', 'ewm_form_points_away', 'days_rest_away'
]

def fetch_real_rolling_stats(match_id):
    """Fetches real rolling stats for a specific fixture from the API."""
    headers = {"x-apisports-key": API_KEY}
    
    # 1. Get Fixture Details
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        
        def get_team_ewm_stats(team_id, span=5):
            # Get last 10 matches to have enough data for a good EWM
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            games = requests.get(url, headers=headers).json().get('response', [])
            
            if not games:
                return 0, 0, 0, 0
            
            stats_list = []
            for g in reversed(games): # EWM expects chronological order
                is_home = g['teams']['home']['id'] == team_id
                gf = g['goals']['home'] if is_home else g['goals']['away']
                ga = g['goals']['away'] if is_home else g['goals']['home']
                gd = gf - ga
                pts = 3 if gd > 0 else (1 if gd == 0 else 0)
                stats_list.append({'gf': gf, 'ga': ga, 'gd': gd, 'pts': pts})
            
            team_df = pd.DataFrame(stats_list)
            ewm = team_df.ewm(span=span).mean().iloc[-1]
            return ewm['gf'], ewm['ga'], ewm['gd'], ewm['pts']

        h_gf, h_ga, h_gd, h_pts = get_team_ewm_stats(h_id)
        a_gf, a_ga, a_gd, a_pts = get_team_ewm_stats(a_id)
        
        return {
            'league_id': match['league']['id'],
            'venue': hash(match['fixture']['venue']['name'] or "Unknown") % 1000,
            'referee': hash(match['fixture']['referee'] or "Unknown") % 1000,
            'ewm_goals_for_home': h_gf, 'ewm_goals_against_home': h_ga, 'ewm_goal_diff_home': h_gd, 'ewm_form_points_home': h_pts, 'days_rest_home': 7,
            'ewm_goals_for_away': a_gf, 'ewm_goals_against_away': a_ga, 'ewm_goal_diff_away': a_gd, 'ewm_form_points_away': a_pts, 'days_rest_away': 7
        }
    except Exception as e:
        print(f"Error fetching stats for match {match_id}: {e}")
        return None

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10}")
    print("-" * 55)
    
    for mid in match_ids:
        data = fetch_real_rolling_stats(mid)
        if data:
            X = pd.DataFrame([data])[FEATURES]
            probs = model.predict_proba(X)[0]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}%")
        else:
            print(f"{mid:<10} | Match data not found or error occurred.")

if __name__ == "__main__":
    ids_file = 'current_progol_ids.json'
    
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            PROGOL_MATCH_IDS = json.load(f)
        print(f"Loaded {len(PROGOL_MATCH_IDS)} match IDs from {ids_file}")
    else:
        print(f"Warning: {ids_file} not found. No IDs to predict.")
        PROGOL_MATCH_IDS = []

    if PROGOL_MATCH_IDS:
        predict_progol(PROGOL_MATCH_IDS)
    else:
        print("No match IDs available to predict.")
