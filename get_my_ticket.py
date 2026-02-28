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
IDS_FILE = 'current_progol_ids.json'

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

def generate_predictions(match_ids, p_model, u_model, scaler, encoder, features):
    headers = {"x-apisports-key": os.getenv('FOOTBALL_API_KEY')}
    probs_list, matches_info = [], []
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            m = res['response'][0]
            h_name, a_name = m['teams']['home']['name'], m['teams']['away']['name']
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            h, a = get_db_team_stats(h_id), get_db_team_stats(a_id)
            data = {'roll_gf_diff': h[0]-a[0], 'roll_ga_diff': h[1]-a[1], 'roll_sh_diff': h[2]-a[2], 'off_eff_diff': h[3]-a[3], 'league_id': m['league']['id']}
            enc_vals = encoder.transform(pd.DataFrame([{'venue': m['fixture']['venue']['name'], 'referee': m['fixture']['referee']}]))
            data['venue_enc'], data['ref_enc'] = enc_vals['venue'].values[0], enc_vals['referee'].values[0]
            X = pd.DataFrame([data])
            for col in features:
                if col not in X.columns: X[col] = 0
            X_scaled = pd.DataFrame(scaler.transform(X[features]), columns=features)
            probs = p_model.predict_proba(X_scaled)[0]
            probs_list.append(probs)
            matches_info.append(f"{h_name} vs {a_name}")
        except: continue
    return probs_list, matches_info

def main():
    print("\n" + "="*50)
    print("🏟️  PROGOL CUSTOM TICKET TOOL  🏟️")
    print("="*50)
    
    if not os.path.exists(PRIMARY_PATH):
        print("❌ Model missing."); return

    # Choice of input mode
    print("Select Input Mode:")
    print("  [1] Optimize by Total Budget (MXN)")
    print("  [2] Fixed count of Doubles & Triples")
    choice = input("Choice: ")

    p_pkg = joblib.load(PRIMARY_PATH); u_pkg = joblib.load(UNDERDOG_PATH)
    p_model, scaler, encoder, features = p_pkg['model'], p_pkg['scaler'], p_pkg['encoder'], p_pkg['features']
    with open(IDS_FILE, 'r') as f: ids = json.load(f).get('match_ids', [])

    print(f"\n🧠 Processing {len(ids)} matches...")
    probs, matches = generate_predictions(ids, p_model, u_pkg['model'], scaler, encoder, features)
    
    if choice == '1':
        try: budget = float(input("Enter budget (MXN): "))
        except: budget = 1000.0
        config, cost, d, t = progol_optimizer.optimize_progol_ticket(probs, budget=budget)
    else:
        try:
            t = int(input("How many TRIPLES? "))
            d = int(input("How many DOUBLES? "))
        except:
            t, d = 0, 0
        config, cost = progol_optimizer.get_custom_ticket_config(probs, d, t)

    print("\n" + "*"*20 + " YOUR CUSTOM TICKET " + "*"*20)
    print(f"Structure: {t} Triples, {d} Doubles | Final Cost: ${cost} MXN")
    progol_optimizer.print_final_ticket(matches, probs, config)
    print("\n✅ TICKET COMPLETE. GOOD LUCK!")

if __name__ == "__main__":
    main()
