import pandas as pd
import numpy as np
import os
import logging
import config
import database
import features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def form_to_points(form_str):
    """Converts a form string like 'WDLWW' into a momentum score."""
    if not form_str or not isinstance(form_str, str): return 0.5
    points = 0
    weight = 1.0
    for char in reversed(form_str[-5:]): # Last 5 games
        if char == 'W': points += 3 * weight
        elif char == 'D': points += 1 * weight
        weight *= 0.9 # Slightly decay older games
    return points

def calculate_alpha_features(df):
    logging.info("🚀 Applying Strategy 10: Strategic Context Engine...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Elo & SoS History
    df, history_df = features.calculate_elo_ratings(df)
    
    # 2. Advanced Rolling Features (EWMA & SoS)
    df = features.add_rolling_features(df, history_df)
    
    # xG Signals
    h_xg = df[['fixture_id', 'date', 'home_id', 'home_xg']].rename(columns={'home_id': 'team_id', 'home_xg': 'xg'})
    a_xg = df[['fixture_id', 'date', 'away_id', 'away_xg']].rename(columns={'away_id': 'team_id', 'away_xg': 'xg'})
    xg_stats = pd.concat([h_xg, a_xg]).sort_values(['team_id', 'date'])
    xg_stats['roll_xg'] = xg_stats.groupby('team_id')['xg'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    df = df.merge(xg_stats[['fixture_id', 'team_id', 'roll_xg']], left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'], suffixes=('', '_h')).drop(columns=['team_id'])
    df = df.merge(xg_stats[['fixture_id', 'team_id', 'roll_xg']], left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'], suffixes=('_h', '_a')).drop(columns=['team_id'])
    
    # 🛑 FIX: Prevent division by zero/inf in Odds
    for col in ['odds_home', 'odds_draw', 'odds_away']:
        df[col] = df[col].replace(0, np.nan)
    
    df['prob_market_h'] = (1 / df['odds_home']).fillna(0.45)
    df['prob_market_d'] = (1 / df['odds_draw']).fillna(0.25)
    df['prob_market_a'] = (1 / df['odds_away']).fillna(0.30)
    
    # Strategic Context Differentials
    df['xg_diff'] = df['roll_xg_h'] - df['roll_xg_a']
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    
    # New: EWMA Differentials (Momentum)
    df['gf_ewma_diff'] = df['home_gf_ewma'] - df['away_gf_ewma']
    df['ga_ewma_diff'] = df['home_ga_ewma'] - df['away_ga_ewma']
    df['sf_ewma_diff'] = df['home_sf_ewma'] - df['away_sf_ewma']
    
    # New: SoS Adjusted Differentials (Quality)
    df['sos_gf_diff'] = df['home_w_gf_ewma'] - df['away_w_gf_ewma']
    
    # New: Rank Gap (Lower rank is better, so Away-Home)
    df['rank_gap'] = df['away_rank'].fillna(15) - df['home_rank'].fillna(15)
    
    # New: Momentum from Form Strings
    df['momentum_h'] = df['home_form'].apply(form_to_points)
    df['momentum_a'] = df['away_form'].apply(form_to_points)
    df['momentum_diff'] = df['momentum_h'] - df['momentum_a']
    
    # New: H2H Dominance
    df['h2h_diff'] = df['h2h_home_wins'].fillna(0) - df['h2h_away_wins'].fillna(0)
    
    # New: Venue Surface Encoding
    df['is_artificial'] = df['venue_surface'].apply(lambda x: 1 if x and 'artificial' in x.lower() else 0)
    
    df['target'] = df.apply(get_target, axis=1)
    
    final_cols = [
        'fixture_id', 'date', 'target', 'league_id', 'venue', 'referee',
        'xg_diff', 'elo_diff', 'rank_gap', 'momentum_diff', 'h2h_diff', 'is_artificial',
        'gf_ewma_diff', 'ga_ewma_diff', 'sf_ewma_diff', 'sos_gf_diff',
        'prob_market_h', 'prob_market_d', 'prob_market_a'
    ]
    
    processed = df[final_cols].fillna(0)
    processed = processed.replace([np.inf, -np.inf], 0)
    
    return processed

def process_matches_from_db():
    df = database.get_all_matches_df()
    logging.info(f"Processing {len(df)} matches from database.")
    return calculate_alpha_features(df)

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info("SUCCESS: Alpha Preprocessing complete.")
