import os
import json
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
import progol_optimizer
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

MODEL_A_PATH = 'models/binary_home_detector.pkl'
MODEL_B_PATH = 'models/draw_away_separator.pkl'
SCALER_PATH = 'models/scaler.pkl'
DB_PATH = 'data/progol.db'

def get_db_team_stats(team_id):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT goals_home, goals_away, home_shots, away_shots, home_id FROM matches WHERE (home_id = ? OR away_id = ?) AND status = 'FT' ORDER BY date DESC LIMIT 5"
    df = pd.read_sql_query(query, conn, params=(team_id, team_id))
    conn.close()
    if df.empty: return 0, 0, 0, 0
    stats = []
    for _, r in df.iterrows():
        is_h = r['home_id'] == team_id
        stats.append({'gf': r['goals_home'] if is_h else r['goals_away'], 'ga': r['goals_away'] if is_h else r['goals_home'], 'sh': r['home_shots'] if is_h else r['away_shots']})
    df_avg = pd.DataFrame(stats).mean()
    return df_avg['gf'], df_avg['ga'], df_avg['sh'], (df_avg['gf']/(df_avg['sh']+1))

def predict_progol(match_ids):
    if not os.path.exists(MODEL_A_PATH): return
    model_a = joblib.load(MODEL_A_PATH)
    model_b = joblib.load(MODEL_B_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open('models/metrics.json', 'r') as f: features = json.load(f)['features']

    import requests
    headers = {"x-apisports-key": os.getenv('FOOTBALL_API_KEY')}
    all_match_probs = []
    final_ids = []

    print(f"\n🚀 CASCADING INFERENCE (Home Filter -> Draw/Away Separator)...")
    
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            m = res['response'][0]
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            
            h = get_db_team_stats(h_id)
            a = get_db_team_stats(a_id)
            
            data = {
                'roll_gf_diff': h[0]-a[0], 'roll_ga_diff': h[1]-a[1], 'roll_sh_diff': h[2]-a[2],
                'off_eff_diff': h[3]-a[3], 'league_id': m['league']['id']
            }
            
            X = pd.DataFrame([data])
            for col in features:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[features]), columns=features)
            
            # --- CASCADING LOGIC ---
            # 1. Ask Model A: "Is it a Home Win?"
            home_prob = model_a.predict_proba(X_scaled)[0][1] # Probability of class 1 (Home)
            
            if home_prob > 0.55: # Confidence threshold for Home
                # We stick with Home prediction
                final_prob = np.array([home_prob, (1-home_prob)*0.4, (1-home_prob)*0.6])
            else:
                # 2. Confidence is low. Let Model B decide between Draw and Away
                b_probs = model_b.predict_proba(X_scaled)[0] # Probs for Draw (1) and Away (2)
                # b_probs might be [P(1), P(2)], we map it back to [P(0), P(1), P(2)]
                final_prob = np.array([home_prob * 0.5, b_probs[0], b_probs[1]])
                final_prob = final_prob / np.sum(final_prob) # Normalize

            all_match_probs.append(final_prob)
            final_ids.append(mid)
            print(f"✅ Resolved {mid} (Home Confidence: {home_prob*100:.1f}%)")
        except: continue

    if all_match_probs:
        config, cost, d, t = progol_optimizer.optimize_progol_ticket(all_match_probs, budget=2000)
        progol_optimizer.print_final_ticket(final_ids, all_match_probs, config)

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
