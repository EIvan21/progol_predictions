import pandas as pd
import numpy as np
import os
import logging
from category_encoders import TargetEncoder
import config
import database
import features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_alpha_features(df):
    logging.info("🚀 Applying Strategy 9: Market Alpha & xG Engine...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Base Signals (Elo & Rolling)
    df = features.calculate_elo_ratings(df)
    df = features.add_rolling_features(df)
    
    # 2. xG Signals (Rolling averages of our new xG column)
    h_xg = df[['fixture_id', 'date', 'home_id', 'home_xg']].rename(columns={'home_id': 'team_id', 'home_xg': 'xg'})
    a_xg = df[['fixture_id', 'date', 'away_id', 'away_xg']].rename(columns={'away_id': 'team_id', 'away_xg': 'xg'})
    xg_stats = pd.concat([h_xg, a_xg]).sort_values(['team_id', 'date'])
    xg_stats['roll_xg'] = xg_stats.groupby('team_id')['xg'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    # Merge xG back
    df = df.merge(xg_stats[['fixture_id', 'team_id', 'roll_xg']], left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'], suffixes=('', '_h')).drop(columns=['team_id'])
    df = df.merge(xg_stats[['fixture_id', 'team_id', 'roll_xg']], left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'], suffixes=('_h', '_a')).drop(columns=['team_id'])
    
    # 3. Market Differentials
    # We turn Odds into Probabilities: Prob = 1/Odd
    df['prob_market_h'] = (1 / df['odds_home']).fillna(0.4)
    df['prob_market_d'] = (1 / df['odds_draw']).fillna(0.25)
    df['prob_market_a'] = (1 / df['odds_away']).fillna(0.35)
    
    # 4. Final Differentials
    df['xg_diff'] = df['roll_xg_h'] - df['roll_xg_a']
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    
    # Context
    df['target'] = df.apply(get_target, axis=1)
    encoder = TargetEncoder(cols=['venue', 'referee'])
    df[['venue_enc', 'ref_enc']] = encoder.fit_transform(df[['venue', 'referee']], df['target'])
    
    final_cols = [
        'fixture_id', 'date', 'target', 'league_id', 
        'xg_diff', 'elo_diff', 'prob_market_h', 'prob_market_d', 'prob_market_a',
        'venue_enc', 'ref_enc'
    ]
    
    return df[final_cols].fillna(0)

def process_matches_from_db():
    df = database.get_all_matches_df()
    # Ensure we only use data that has been enriched
    df = df.dropna(subset=['odds_home'])
    logging.info(f"Processing {len(df)} High-Signal Matches.")
    return calculate_alpha_features(df)

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info("SUCCESS: Alpha Preprocessing complete.")
