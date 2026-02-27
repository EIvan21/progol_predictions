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

def fetch_match_stats(team_id):
    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
    try:
        data = requests.get(url, headers=headers).json().get('response', [])
        if not data: return 0, 0, 0, 50, 4 # gf, ga, sh, po, co
        stats = []
        for g in reversed(data):
            is_h = g['teams']['home']['id'] == team_id
            gf, ga = (g['goals']['home'], g['goals']['away']) if is_h else (g['goals']['away'], g['goals']['home'])
            stats.append({'gf':gf, 'ga':ga, 'sh': 10, 'po': 50, 'co': 4})
        df = pd.DataFrame(stats).mean()
        
        # Calculate Interaction Terms
        off_eff = df['gf'] / (df['sh'] + 1)
        press_idx = (df['po'] * df['co']) / 100
        def_res = df['sh'] / (df['ga'] + 1)
        
        return df['gf'], df['ga'], df['sh'], df['po'], df['co'], off_eff, press_idx, def_res
    except: return 0, 0, 0, 50, 4, 0, 2, 0

def predict_progol(match_ids):
    if not os.path.exists(METRICS_PATH): return
    with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
    FEATURES = metrics['features']
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)

    headers = {"x-apisports-key": API_KEY}
    print(f"\n{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10} | {'PRED':<5}")
    print("-" * 65)
    
    for mid in match_ids:
        try:
            m_res = requests.get(f"{BASE_URL}/fixtures?id={mid}", headers=headers).json()['response'][0]
            h_id, a_id = m_res['teams']['home']['id'], m_res['teams']['away']['id']
            
            h = fetch_match_stats(h_id)
            a = fetch_match_stats(a_id)
            
            # Map stats to the new Differential Interface
            data = {
                'league_id': m_res['league']['id'],
                'venue_encoded': 0.45, 'ref_encoded': 0.33,
                'roll_gf_diff': h[0] - a[0],
                'roll_ga_diff': h[1] - a[1],
                'roll_sh_diff': h[2] - a[2],
                'roll_po_diff': h[3] - a[3],
                'roll_co_diff': h[4] - a[4],
                'off_efficiency_diff': h[5] - a[5],
                'pressure_index_diff': h[6] - a[6],
                'def_resilience_diff': h[7] - a[7]
            }
            
            X = pd.DataFrame([data])
            for col in FEATURES:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
            probs = model.predict_proba(X_scaled)[0]
            pred_label = {0:'L', 1:'E', 2:'V'}[np.argmax(probs)]
            print(f"{mid:<10} | {probs[0]*100:8.2f}% | {probs[1]*100:8.2f}% | {probs[2]*100:8.2f}% |  {pred_label}")
        except: continue

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
