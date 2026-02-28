import os
import json
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
import progol_optimizer
import warnings
import requests

warnings.filterwarnings("ignore")
load_dotenv()

PRIMARY_PATH = 'models/calibrated_ensemble.pkl'
UNDERDOG_PATH = 'models/underdog_specialist.pkl'
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
    if not os.path.exists(PRIMARY_PATH): return
    p_pkg = joblib.load(PRIMARY_PATH)
    u_pkg = joblib.load(UNDERDOG_PATH)
    
    p_model, scaler, encoder, features = p_pkg['model'], p_pkg['scaler'], p_pkg['encoder'], p_pkg['features']
    u_model = u_pkg['model']

    headers = {"x-apisports-key": os.getenv('FOOTBALL_API_KEY')}
    
    table_primary = []
    table_underdog = []

    print(f"\n🚀 GENERATING DUAL-PERSPECTIVE ANALYSIS...")
    
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            m = res['response'][0]
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            h = get_db_team_stats(h_id)
            a = get_db_team_stats(a_id)
            
            # Differential Data
            data = {
                'roll_gf_diff': h[0]-a[0], 'roll_ga_diff': h[1]-a[1], 'roll_sh_diff': h[2]-a[2],
                'off_eff_diff': h[3]-a[3], 'league_id': m['league']['id']
            }
            
            # Categories
            enc_vals = encoder.transform(pd.DataFrame([{'venue': m['fixture']['venue']['name'], 'referee': m['fixture']['referee']}]))
            data['venue_enc'] = enc_vals['venue'].values[0]
            data['ref_enc'] = enc_vals['referee'].values[0]
            
            X = pd.DataFrame([data])
            for col in features:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[features]), columns=features)
            
            # 1. PRIMARY PREDICTION (Home/Draw/Away)
            p_probs = p_model.predict_proba(X_scaled)[0]
            table_primary.append({'id': mid, 'h': p_probs[0], 'd': p_probs[1], 'a': p_probs[2]})
            
            # 2. UNDERDOG FILTER (If not Home, which is better?)
            u_probs = u_model.predict_proba(X_scaled)[0] # Probs for 1 (Draw) and 2 (Away)
            table_underdog.append({'id': mid, 'd_vs_a': 'Draw' if u_probs[0] > u_probs[1] else 'Away', 'd_score': u_probs[0], 'a_score': u_probs[1]})
            
            print(f"✅ Analyzed {mid}")
        except: continue

    # --- TABLE 1: SCIENTIFIC PROBABILITIES ---
    print("\n" + "="*25 + " TABLE 1: SCIENTIFIC PROBABILITIES " + "="*25)
    print(f"{'Match ID':<10} | {'Home (%)':<10} | {'Draw (%)':<10} | {'Away (%)':<10} | {'PRED'}")
    print("-" * 65)
    for r in table_primary:
        p_idx = np.argmax([r['h'], r['d'], r['a']])
        label = {0:'L', 1:'E', 2:'V'}[p_idx]
        print(f"{r['id']:<10} | {r['h']*100:8.2f}% | {r['d']*100:8.2f}% | {r['a']*100:8.2f}% |  {label}")

    # --- TABLE 2: THE UNDERDOG FILTER (E vs V) ---
    print("\n" + "="*25 + " TABLE 2: DRAW VS AWAY SPECIALIST " + "="*25)
    print(f"{'Match ID':<10} | {'Draw Conf':<10} | {'Away Conf':<10} | {'STRONGEST UNDERDOG'}")
    print("-" * 65)
    for r in table_underdog:
        print(f"{r['id']:<10} | {r['d_score']*100:8.2f}% | {r['a_score']*100:8.2f}% |  {r['d_vs_a']}")

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
