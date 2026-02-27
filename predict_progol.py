import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle
import joblib
import progol_optimizer

load_dotenv()
MODEL_PATH = 'models/calibrated_ensemble.pkl'
IDS_FILE = 'current_progol_ids.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_match_stats(team_id):
    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}/fixtures?team={team_id}&last=5&status=FT"
    try:
        data = requests.get(url, headers=headers).json().get('response', [])
        if not data: return 0, 0, 0, 0
        stats = []
        for g in data:
            is_h = g['teams']['home']['id'] == team_id
            gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
            stats.append({'gf':gf, 'ga':ga, 'sh': 10, 'po': 50})
        df = pd.DataFrame(stats).mean()
        return df['gf'], df['ga'], df['sh'], df['po']
    except: return 0, 0, 5, 50

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH): return
    package = joblib.load(MODEL_PATH)
    model, scaler, features = package['model'], package['scaler'], package['features']

    all_match_probs = []
    processed_ids = []

    print(f"\n🚀 GENERATING OPTIMIZED TICKET (Budget: $2,000 MXN)")
    print(f"Analyzing {len(match_ids)} matches...")

    for mid in match_ids:
        # Mocking context fetch for architecture demonstration
        # In full run, this calls the API
        h_stats = fetch_match_stats(1) # Placeholder
        a_stats = fetch_match_stats(2) # Placeholder
        
        data = {
            'league_id': 262, 'elo_prob_h': 0.45, 'elo_diff': 0,
            'roll_form_diff': 0, 'roll_gf_diff': 0, 'roll_ga_diff': 0,
            'roll_sf_diff': 0, 'roll_sa_diff': 0, 'venue_enc': 0.45, 'ref_enc': 0.33
        }
        
        df_in = pd.DataFrame([data])
        for col in features:
            if col not in df_in.columns: df_in[col] = 0
        
        X_scaled = scaler.transform(df_in[features])
        probs = model.predict_proba(X_scaled)[0]
        
        # Apply Bayesian Correction (optional, implemented in engine)
        all_match_probs.append(probs)
        processed_ids.append(mid)

    # 3. OPTIMIZE TICKET
    if len(all_match_probs) >= 14:
        # Progol only uses 14 matches
        final_probs = all_match_probs[:14]
        final_ids = processed_ids[:14]
    else:
        # Pad with neutral if less than 14
        needed = 14 - len(all_match_probs)
        final_probs = all_match_probs + [np.array([0.33, 0.33, 0.34])] * needed
        final_ids = processed_ids + [0] * needed

    config, cost, d, t = progol_optimizer.optimize_progol_ticket(final_probs, budget=2000)
    
    print(f"\n--- OPTIMAL TICKET STRUCTURE ---")
    print(f"Configuration: {d} Doubles, {t} Triples")
    print(f"Total Cost: ${cost} MXN")
    
    progol_optimizer.print_final_ticket(final_ids, final_probs, config)

if __name__ == "__main__":
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE, 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
        predict_progol(ids)
