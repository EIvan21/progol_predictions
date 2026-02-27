import os
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pickle
import joblib
import progol_optimizer
import warnings
import time

warnings.filterwarnings("ignore")
load_dotenv()

MODEL_PATH = 'models/calibrated_ensemble.pkl'
IDS_FILE = 'current_progol_ids.json'
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"

def fetch_live_match_stats(match_id):
    """Fetches real-time pre-match stats from the API."""
    headers = {"x-apisports-key": API_KEY}
    try:
        # 1. Get Match Info
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        lid = match['league']['id']
        
        def get_team_rolling(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0, 0 # pts, gf, ga, sf, sa
            stats = []
            for g in data:
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                sf = 5 # Default shot proxy if missing
                stats.append({'pts': (3 if gf>ga else (1 if gf==ga else 0)), 'gf':gf, 'ga':ga, 'sf':sf})
            return pd.DataFrame(stats).mean().tolist()

        h_stats = get_team_rolling(h_id)
        time.sleep(0.5) # Rate limit
        a_stats = get_team_rolling(a_id)
        
        # Must match feature order in preprocess.py
        return {
            'league_id': lid,
            'elo_prob_h': 0.45, # Elo would require historical database lookup
            'elo_diff': 0,
            'roll_form_diff': h_stats[0] - a_stats[0],
            'roll_gf_diff': h_stats[1] - a_stats[1],
            'roll_ga_diff': h_stats[2] - a_stats[2],
            'roll_sf_diff': h_stats[3] - a_stats[3],
            'roll_sa_diff': h_stats[3] - a_stats[3], # Approximate
            'venue_enc': 0.45, 'ref_enc': 0.33
        }
    except Exception as e:
        print(f"DEBUG: Error fetching match {match_id}: {e}")
        return None

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train it first.")
        return
        
    package = joblib.load(MODEL_PATH)
    model, scaler, features = package['model'], package['scaler'], package['features']

    all_match_probs = []
    final_ids = []

    print(f"\n🚀 ANALYZING LIVE DATA FOR {len(match_ids)} MATCHES...")
    
    for mid in match_ids:
        data = fetch_live_match_stats(mid)
        if data:
            df_in = pd.DataFrame([data])
            # Ensure order and existence
            for col in features:
                if col not in df_in.columns: df_in[col] = 0
            
            X_scaled = pd.DataFrame(scaler.transform(df_in[features]), columns=features)
            probs = model.predict_proba(X_scaled)[0]
            
            all_match_probs.append(probs)
            final_ids.append(mid)
            print(f"✅ Resolved context for match {mid}")
        else:
            print(f"❌ Failed to resolve context for match {mid}")

    # Pad if necessary to reach 14
    if len(all_match_probs) < 14:
        needed = 14 - len(all_match_probs)
        all_match_probs += [np.array([0.45, 0.25, 0.30])] * needed # Historical priors
        final_ids += [0] * needed

    # Optimize Budget
    config, cost, d, t = progol_optimizer.optimize_progol_ticket(all_match_probs, budget=2000)
    
    print(f"\n--- 💰 OPTIMIZED PROGOL TICKET ---")
    print(f"Budget Utilization: ${cost} / $2,000 MXN")
    print(f"Configuration: {d} Doubles, {t} Triples")
    
    progol_optimizer.print_final_ticket(final_ids, all_match_probs, config)

if __name__ == "__main__":
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE, 'r') as f:
            cache = json.load(f)
            ids = cache.get('match_ids', []) if isinstance(cache, dict) else cache
        predict_progol(ids)
