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
    probs_list = []
    matches_info = []
    
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
            
            probs = p_model.predict_proba(X_scaled)[0]
            probs_list.append(probs)
            matches_info.append(f"{h_name} vs {a_name}")
        except: continue
    return probs_list, matches_info

def main():
    print("
" + "="*50)
    print("🏟️  PROGOL INSTANT TICKET OPTIMIZER  🏟️")
    print("="*50)
    
    if not os.path.exists(PRIMARY_PATH):
        print("❌ Error: No trained model found. Run pipeline first.")
        return

    # 1. User Budget Input
    try:
        budget = float(input("
Enter your budget for the 14-game ticket (MXN): "))
    except:
        budget = 500.0
        print(f"Invalid input. Using default budget: ${budget}")

    # 2. Load Models
    p_pkg = joblib.load(PRIMARY_PATH)
    u_pkg = joblib.load(UNDERDOG_PATH)
    p_model, scaler, encoder, features = p_pkg['model'], p_pkg['scaler'], p_pkg['encoder'], p_pkg['features']
    
    # 3. Load Current Slate
    if not os.path.exists(IDS_FILE):
        print("❌ Error: No current Progol IDs found. Run pipeline first.")
        return
    with open(IDS_FILE, 'r') as f:
        ids = json.load(f).get('match_ids', [])

    # 4. Predict Progol (14 matches)
    print(f"
🧠 Calculating probabilities for Progol (14 matches)...")
    progol_ids = ids[:14]
    probs, matches = generate_predictions(progol_ids, p_model, u_pkg['model'], scaler, encoder, features)
    
    # Pad to 14 if scraper found fewer
    while len(probs) < 14:
        probs.append(np.array([0.45, 0.25, 0.30]))
        matches.append("Match Not Resolved")

    # 5. Optimize & Display
    config, cost, d, t = progol_optimizer.optimize_progol_ticket(probs, budget=budget)
    
    print("
" + "*"*20 + " YOUR OPTIMIZED TICKET " + "*"*20)
    print(f"Budget: ${budget} | Real Cost: ${cost} | Structure: {t}T, {d}D, {14-t-d}S")
    progol_optimizer.print_final_ticket(progol_ids + [0]*(14-len(progol_ids)), probs, config)

    # 6. Revancha (7 matches) - Fixed budget strategy ($5 per simple line)
    # Note: Revancha has different mechanics, but we can provide the raw 3-way picks.
    if len(ids) > 14:
        print("
" + "="*20 + " REVANCHA (7 MATCHES) " + "="*20)
        revancha_ids = ids[14:21]
        r_probs, r_matches = generate_predictions(revancha_ids, p_model, u_pkg['model'], scaler, encoder, features)
        for i, (m, p) in enumerate(zip(r_matches, r_probs)):
            label = {0:'L', 1:'E', 2:'V'}[np.argmax(p)]
            print(f"R{i+1}: {m:<30} | Best Pick: {label} ({np.max(p)*100:.1f}%)")

    print("
✅ TICKET GENERATION COMPLETE. GOOD LUCK!")

if __name__ == "__main__":
    main()
