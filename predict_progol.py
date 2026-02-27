import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle
import warnings
import time

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_hyper_context(match_id):
    headers = {"x-apisports-key": API_KEY}
    try:
        # 1. Get Fixture Info
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'):
            print(f"DEBUG: No API response for match {match_id}")
            return None
            
        match = res['response'][0]
        h_id = match['teams']['home']['id']
        a_id = match['teams']['away']['id']
        
        def get_team_stats(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=5&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0
            
            stats = []
            for g in data:
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                stats.append({'gf': gf or 0, 'ga': ga or 0, 'sh': 5, 'po': 50})
            
            df = pd.DataFrame(stats).mean()
            return df['gf'], df['ga'], df['sh'], df['po']

        h_gf, h_ga, h_sh, h_po = get_team_stats(h_id)
        time.sleep(0.5) # Rate limit safety
        a_gf, a_ga, a_sh, a_po = get_team_stats(a_id)
        
        return {
            'league_id': match['league']['id'],
            'venue_encoded': 0.45, 'ref_encoded': 0.33,
            'roll_gf_diff': h_gf - a_gf,
            'roll_ga_diff': h_ga - a_ga,
            'roll_sh_diff': h_sh - a_sh,
            'roll_po_diff': h_po - a_po,
            'roll_co_diff': 0, 'off_efficiency_diff': 0, 'pressure_index_diff': 0, 'def_resilience_diff': 0
        }
    except Exception as e:
        print(f"DEBUG Error for {match_id}: {str(e)}")
        return None

def predict_progol(match_ids):
    MODEL_PATH = 'models/progol_stack_model.bin'
    SCALER_PATH = 'models/scaler.pkl'
    METRICS_PATH = 'models/metrics.json'

    if not os.path.exists(MODEL_PATH):
        print("Error: Model missing.")
        return
        
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics.get('features', [])
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)

    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10} | {'PRED':<5}")
    print("-" * 65)
    
    for mid in match_ids:
        data = fetch_hyper_context(mid)
        if data:
            df_in = pd.DataFrame([data])
            for col in FEATURES:
                if col not in df_in.columns: df_in[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(df_in[FEATURES]), columns=FEATURES)
            probs = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_label = {0: 'L', 1: 'E', 2: 'V'}[pred_idx]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}% |  {pred_label}")
        time.sleep(1) # Final safety

if __name__ == "__main__":
    IDS_FILE = 'current_progol_ids.json'
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE, 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
            predict_progol(ids)
