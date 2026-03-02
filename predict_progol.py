import os
import json
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
import warnings
import requests
import logging

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PRIMARY_PATH = 'models/calibrated_ensemble.pkl'
DB_PATH = 'data/progol.db'

def form_to_points(form_str):
    if not form_str or not isinstance(form_str, str): return 0.5
    points = 0
    weight = 1.0
    for char in reversed(form_str[-5:]):
        if char == 'W': points += 3 * weight
        elif char == 'D': points += 1 * weight
        weight *= 0.9
    return points

def get_inference_data(h_id, a_id, lid, season):
    """Fetches the latest strategic stats for inference."""
    conn = sqlite3.connect(DB_PATH)
    
    def get_team_context(tid):
        # Get latest rank and form from matches table
        q = "SELECT home_rank, home_form FROM matches WHERE home_id = ? AND home_rank IS NOT NULL ORDER BY date DESC LIMIT 1"
        res = pd.read_sql_query(q, conn, params=(tid,))
        if res.empty:
            q = "SELECT away_rank, away_form FROM matches WHERE away_id = ? AND away_rank IS NOT NULL ORDER BY date DESC LIMIT 1"
            res = pd.read_sql_query(q, conn, params=(tid,))
            if res.empty: return 10, "DDDDD"
            return res.iloc[0]['away_rank'], res.iloc[0]['away_form']
        return res.iloc[0]['home_rank'], res.iloc[0]['home_form']

    def get_rolling_stats(tid):
        # Get last 5 games for EWMA-like average
        q = """
            SELECT goals_home as gf, goals_away as ga, home_shots as sf, home_xg as xg, home_id FROM matches 
            WHERE (home_id = ? OR away_id = ?) AND status = 'FT' 
            ORDER BY date DESC LIMIT 5
        """
        df = pd.read_sql_query(q, conn, params=(tid, tid))
        if df.empty: return 0, 0, 0, 0
        
        # Calculate simple means for inference (proxy for EWMA)
        return df['gf'].mean(), df['ga'].mean(), df['sf'].mean(), df['xg'].mean()

    h_rank, h_form = get_team_context(h_id)
    a_rank, a_form = get_team_context(a_id)
    h_stats = get_rolling_stats(h_id)
    a_stats = get_rolling_stats(a_id)
    
    # H2H
    q = "SELECT goals_home, goals_away FROM matches WHERE (home_id = ? AND away_id = ?) OR (home_id = ? AND away_id = ?) LIMIT 10"
    h2h = pd.read_sql_query(q, conn, params=(h_id, a_id, a_id, h_id))
    h2h_diff = 0
    if not h2h.empty:
        h2h_diff = (h2h['goals_home'] > h2h['goals_away']).sum() - (h2h['goals_home'] < h2h['goals_away']).sum()

    conn.close()
    
    return {
        'xg_diff': h_stats[3] - a_stats[3],
        'elo_diff': 0, # Placeholder if not using live Elo
        'rank_gap': a_rank - h_rank,
        'momentum_diff': form_to_points(h_form) - form_to_points(a_form),
        'h2h_diff': h2h_diff,
        'is_artificial': 0, # Default
        'gf_ewma_diff': h_stats[0] - a_stats[0],
        'ga_ewma_diff': h_stats[1] - a_stats[1],
        'sf_ewma_diff': h_stats[2] - a_stats[2],
        'sos_gf_diff': (h_stats[0] * 1.1) - (a_stats[0] * 1.1) # Proxy
    }

def predict_progol(match_ids):
    if not os.path.exists(PRIMARY_PATH):
        print("❌ Error: Model not found. Run training first.")
        return
        
    pkg = joblib.load(PRIMARY_PATH)
    model, scaler, encoder, features = pkg['model'], pkg['scaler'], pkg['encoder'], pkg['features']
    
    headers = {"x-apisports-key": os.getenv('FOOTBALL_API_KEY')}
    results = []
    
    print(f"\n🚀 ANALYZING PROGOL SLATE ({len(match_ids)} MATCHES)...")
    
    for mid in match_ids:
        try:
            res = requests.get(f"https://v3.football.api-sports.io/fixtures?id={mid}", headers=headers).json()
            if not res.get('response'): continue
            m = res['response'][0]
            h_name, a_name = m['teams']['home']['name'], m['teams']['away']['name']
            h_id, a_id = m['teams']['home']['id'], m['teams']['away']['id']
            lid = m['league']['id']
            season = m['league']['season']
            
            # Fetch Strategic Data
            inf_data = get_inference_data(h_id, a_id, lid, season)
            
            # Add Market Odds
            inf_data['prob_market_h'], inf_data['prob_market_d'], inf_data['prob_market_a'] = 0.33, 0.33, 0.33 # Defaults
            # Try to get live odds
            o_res = requests.get(f"https://v3.football.api-sports.io/odds?fixture={mid}&bookmaker=8", headers=headers).json().get('response', [])
            if o_res and o_res[0].get('bookmakers'):
                bets = o_res[0]['bookmakers'][0]['bets'][0]['values']
                inf_data['prob_market_h'] = 1 / float(bets[0]['odd'])
                inf_data['prob_market_d'] = 1 / float(bets[1]['odd'])
                inf_data['prob_market_a'] = 1 / float(bets[2]['odd'])

            # Add Categoricals for Encoder
            inf_data['league_id'] = lid
            inf_data['venue'] = m['fixture']['venue']['name'] or "Unknown"
            inf_data['referee'] = m['fixture']['referee'] or "Unknown"

            # 1. Encode
            cat_df = pd.DataFrame([inf_data])[['league_id', 'venue', 'referee']]
            encoded_df = encoder.transform(cat_df)
            
            # 2. Build Final Feature Vector
            X_dict = inf_data.copy()
            X_dict.update(encoded_df.iloc[0].to_dict())
            
            X = pd.DataFrame([X_dict])
            for col in features:
                if col not in X.columns: X[col] = 0
            
            X_final = pd.DataFrame(scaler.transform(X[features]), columns=features)
            
            # 3. Predict
            probs = model.predict_proba(X_final)[0]
            
            results.append({
                'match': f"{h_name} vs {a_name}",
                'h': probs[0], 'd': probs[1], 'v': probs[2]
            })
            print(f"✅ Processed: {h_name} vs {a_name}")
        except Exception as e:
            # print(f"⚠️ Skip {mid}: {e}")
            continue

    # --- REPORT ---
    print("\n" + "="*40 + " SCIENTIFIC PROGOL REPORT " + "="*40)
    print(f"{'GAME':<3} | {'MATCHUP':<35} | {'HOME %':<8} | {'DRAW %':<8} | {'AWAY %':<8} | {'PRED'}")
    print("-" * 105)
    
    for i, r in enumerate(results):
        p_idx = np.argmax([r['h'], r['d'], r['v']])
        label = {0:'L', 1:'E', 2:'V'}[p_idx]
        print(f"{i+1:<3} | {r['match']:<35} | {r['h']*100:6.1f}% | {r['d']*100:6.1f}% | {r['v']*100:6.1f}% |  {label}")
    print("="*105 + "\n")

if __name__ == "__main__":
    ids_file = 'current_progol_ids.json'
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            data = json.load(f)
            ids = data.get('match_ids', [])
        predict_progol(ids)
