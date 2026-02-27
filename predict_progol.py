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
    headers = {"x-apisports-key": API_KEY}
    try:
        res = requests.get(f"{BASE_URL}/fixtures?id={match_id}", headers=headers).json()
        if not res.get('response'): return None
        match = res['response'][0]
        h_id, a_id = match['teams']['home']['id'], match['teams']['away']['id']
        lid = match['league']['id']
        
        def get_team_rolling(team_id):
            url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
            data = requests.get(url, headers=headers).json().get('response', [])
            if not data: return 0, 0, 0, 0, 0
            stats = []
            for g in data:
                is_h = g['teams']['home']['id'] == team_id
                gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
                stats.append({'pts': (3 if gf>ga else (1 if gf==ga else 0)), 'gf':gf, 'ga':ga, 'sf':5})
            return pd.DataFrame(stats).mean().tolist()

        h_s = get_team_rolling(h_id)
        time.sleep(0.5)
        a_s = get_team_rolling(a_id)
        
        return {
            'league_id': lid, 'elo_prob_h': 0.45, 'elo_diff': 0,
            'roll_form_diff': h_s[0] - a_s[0], 'roll_gf_diff': h_s[1] - a_s[1],
            'roll_ga_diff': h_s[2] - a_s[2], 'roll_sf_diff': h_s[3] - a_s[3],
            'roll_sa_diff': h_s[3] - a_s[3], 'venue_enc': 0.45, 'ref_enc': 0.33
        }
    except: return None

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH): return
    package = joblib.load(MODEL_PATH)
    model, scaler, imputer, features = package['model'], package['scaler'], package['imputer'], package['features']

    all_match_probs = []
    final_ids = []

    print(f"\n🚀 ANALYZING LIVE DATA FOR {len(match_ids)} MATCHES...")
    
    for mid in match_ids:
        data = fetch_live_match_stats(mid)
        if data:
            df_in = pd.DataFrame([data])
            for col in features:
                if col not in df_in.columns: df_in[col] = 0
            
            # Step 1: Impute missing values (Safe from NaNs)
            X_clean = pd.DataFrame(imputer.transform(df_in[features]), columns=features)
            # Step 2: Scale
            X_scaled = pd.DataFrame(scaler.transform(X_clean), columns=features)
            
            probs = model.predict_proba(X_scaled)[0]
            all_match_probs.append(probs)
            final_ids.append(mid)
            print(f"✅ Resolved {mid}")
        else:
            print(f"❌ Failed {mid}")

    if len(all_match_probs) < 14:
        all_match_probs += [np.array([0.45, 0.25, 0.30])] * (14 - len(all_match_probs))
        final_ids += [0] * (14 - len(final_ids))

    config, cost, d, t = progol_optimizer.optimize_progol_ticket(all_match_probs, budget=2000)
    print(f"\n💰 OPTIMIZED TICKET: ${cost} MXN ({d} Doubles, {t} Triples)")
    progol_optimizer.print_final_ticket(final_ids, all_match_probs, config)

if __name__ == "__main__":
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE, 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
