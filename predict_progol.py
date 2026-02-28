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
    
    print(f"\n🚀 ANALYZING PROGOL SLATE (14 MATCHES)...")
    
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            m = res['response'][0]
            h_name, a_name = m['teams']['home']['name'], m['teams']['away']['name']
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            
            h = get_db_team_stats(h_id)
            a = get_db_team_stats(a_id)
            
            data = {'roll_gf_diff': h[0]-a[0], 'roll_ga_diff': h[1]-a[1], 'roll_sh_diff': h[2]-a[2], 'off_eff_diff': h[3]-a[3], 'league_id': m['league']['id']}
            enc_vals = encoder.transform(pd.DataFrame([{'venue': m['fixture']['venue']['name'], 'referee': m['fixture']['referee']}]))
            data['venue_enc'], data['ref_enc'] = enc_vals['venue'].values[0], enc_vals['referee'].values[0]
            
            X = pd.DataFrame([data])
            for col in features:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[features]), columns=features)
            
            # Get Standard Probs
            probs = p_model.predict_proba(X_scaled)[0]
            # Get Underdog Filter (E vs V)
            u_probs = u_model.predict_proba(X_scaled)[0]
            
            table_primary.append({
                'id': mid, 'match': f"{h_name} vs {a_name}", 
                'h': probs[0], 'd': probs[1], 'v': probs[2],
                'u_d': u_probs[0], 'u_v': u_probs[1]
            })
            print(f"✅ Processed: {h_name} vs {a_name}")
        except: continue

    # --- FINAL REPORT OUTPUT ---
    print("\n" + "="*40 + " SCIENTIFIC PROGOL REPORT " + "="*40)
    print(f"{'GAME':<3} | {'MATCHUP':<35} | {'HOME %':<8} | {'DRAW %':<8} | {'AWAY %':<8} | {'PRED'}")
    print("-" * 105)
    
    all_probs = []
    for i, r in enumerate(table_primary):
        p_idx = np.argmax([r['h'], r['d'], r['v']])
        label = {0:'L', 1:'E', 2:'V'}[p_idx]
        all_probs.append(np.array([r['h'], r['d'], r['v']]))
        print(f"{i+1:<3} | {r['match']:<35} | {r['h']*100:6.1f}% | {r['d']*100:6.1f}% | {r['v']*100:6.1f}% |  {label}")

    # --- OPTIMIZED TICKET CONSTRUCTION ---
    if len(all_probs) >= 11:
        print("\n" + "="*35 + " SUGGESTED PROGOL TICKET " + "="*35)
        # Pad if needed
        while len(all_probs) < 14: all_probs.append(np.array([0.45, 0.25, 0.30]))
        
        # Optimize: Standard high-value play (3 Doubles, 2 Triples)
        # Progol Mexico common combo: 2 Triples + 3 Doubles = $1,080 MXN
        config, cost, d, t = progol_optimizer.optimize_progol_ticket(all_probs, budget=1200)
        
        print(f"Ticket Strategy: {t} Triples, {d} Doubles, {14-t-d} Singles")
        print(f"Estimated Cost: ${cost} MXN")
        print("-" * 95)
        
        for i, (p, c) in enumerate(zip(all_probs, config)):
            label = {0:'L', 1:'E', 2:'V'}[np.argmax(p)]
            if c == 'S': pick = label
            elif c == 'D': 
                top2 = np.argsort(p)[-2:]
                pick = "/".join([{0:'L', 1:'E', 2:'V'}[x] for x in sorted(top2)])
            else: pick = "L/E/V"
            print(f"Game {i+1:2}: [{pick:^7}]  (Confidence: {np.max(p)*100:4.1f}%)")
        print("=" * 95)

if __name__ == "__main__":
    ids_file = 'current_progol_ids.json'
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
