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

MODEL_PATH = 'models/calibrated_ensemble.pkl'
DB_PATH = 'data/progol.db'

def get_db_team_stats(team_id):
    """Pulls historical stats directly from local DB - 100x faster than API."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT goals_home, goals_away, home_shots, away_shots, home_corners, away_corners, home_possession, away_possession, home_id
        FROM matches 
        WHERE (home_id = ? OR away_id = ?) AND status = 'FT'
        ORDER BY date DESC LIMIT 5
    """
    df = pd.read_sql_query(query, conn, params=(team_id, team_id))
    conn.close()
    
    if df.empty: return 0, 0, 0, 50, 4, 0, 2
    
    stats = []
    for _, row in df.iterrows():
        is_h = row['home_id'] == team_id
        gf, ga = (row['goals_home'], row['goals_away']) if is_h else (row['goals_away'], row['goals_home'])
        sf, sa = (row['home_shots'], row['away_shots']) if is_h else (row['away_shots'], row['home_shots'])
        po = row['home_possession'] if is_h else row['away_possession']
        co = row['home_corners'] if is_h else row['away_corners']
        stats.append({'gf':gf, 'ga':ga, 'sh':sf or 5, 'po':po or 50, 'co':co or 4})
    
    df_avg = pd.DataFrame(stats).mean()
    off_eff = df_avg['gf'] / (df_avg['sh'] + 1)
    press_idx = (df_avg['po'] * df_avg['co']) / 100
    
    return df_avg['gf'], df_avg['ga'], df_avg['sh'], df_avg['po'], df_avg['co'], off_eff, press_idx

def predict_progol(match_ids):
    if not os.path.exists(MODEL_PATH): return
    package = joblib.load(MODEL_PATH)
    model, scaler, encoder, features = package['model'], package['scaler'], package['encoder'], package['features']

    # Fetch match metadata from API (only for IDs and names)
    import requests
    headers = {"x-apisports-key": os.getenv('FOOTBALL_API_KEY')}
    
    all_match_probs = []
    final_ids = []

    print(f"\n🚀 SHIELDED INFERENCE STARTING...")
    
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            m = res['response'][0]
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            
            # Fetch from LOCAL DB (Fast!)
            h = get_db_team_stats(h_id)
            a = get_db_team_stats(a_id)
            
            # Differential Calculation
            data = {
                'roll_gf_diff': h[0] - a[0], 'roll_ga_diff': h[1] - a[1],
                'roll_sh_diff': h[2] - a[2], 'roll_po_diff': h[3] - a[3],
                'roll_co_diff': h[4] - a[4], 'off_eff_diff': h[5] - a[5],
                'press_idx_diff': h[6] - a[6], 'league_id': m['league']['id']
            }
            
            # Apply Encoder (Venue/Referee)
            enc_input = pd.DataFrame([{'venue': m['fixture']['venue']['name'], 'referee': m['fixture']['referee']}])
            enc_vals = encoder.transform(enc_input)
            data['venue_enc'] = enc_vals['venue'].values[0]
            data['ref_enc'] = enc_vals['referee'].values[0]
            
            df_in = pd.DataFrame([data])
            for col in features:
                if col not in df_in.columns: df_in[col] = 0
            
            X_scaled = pd.DataFrame(scaler.transform(df_in[features]), columns=features)
            probs = model.predict_proba(X_scaled)[0]
            
            all_match_probs.append(probs)
            final_ids.append(mid)
            print(f"✅ Resolved Match {mid}")
        except Exception as e:
            print(f"❌ Failed {mid}: {e}")

    # Optimize and Print
    if all_match_probs:
        config, cost, d, t = progol_optimizer.optimize_progol_ticket(all_match_probs, budget=2000)
        print(f"\n💰 OPTIMIZED TICKET: ${cost} MXN")
        progol_optimizer.print_final_ticket(final_ids, all_match_probs, config)

if __name__ == "__main__":
    if os.path.exists('current_progol_ids.json'):
        with open('current_progol_ids.json', 'r') as f:
            ids = json.load(f).get('match_ids', [])
        predict_progol(ids)
