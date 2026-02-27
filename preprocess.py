import pandas as pd
import numpy as np
import os
import logging
from category_encoders import TargetEncoder
import config
import database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_differential_features(df):
    logging.info("🚀 Applying Strategy 6: Differential Rivalry Engineering...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Team-Level Processing
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'home_possession', 'home_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'away_possession', 'away_corners']].copy()
    h.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'shots', 'poss', 'corners']
    a.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'shots', 'poss', 'corners']
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    group = team_stats.groupby('team_id')
    
    # Rolling Stats
    for col in ['gf', 'ga', 'shots', 'poss', 'corners']:
        team_stats[f'roll_{col}'] = group[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    team_stats['cs_rate'] = group['ga'].transform(lambda x: (x.shift() == 0).rolling(5, min_periods=1).mean()).fillna(0)
    team_stats['power_score'] = group['gf'].transform(lambda x: (x.shift() > 0).rolling(10, min_periods=1).mean()).fillna(0.3)

    # 2. Merge Back
    match_stats = df.copy()
    for suffix in ['home', 'away']:
        tid = f'{suffix}_id'
        cols = ['fixture_id', 'team_id', 'roll_gf', 'roll_ga', 'roll_shots', 'roll_poss', 'roll_corners', 'cs_rate', 'power_score']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', tid], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    # 3. DIFFERENTIAL ENGINEERING (The Breakthrough)
    # We calculate the GAP between teams. Positive = Home is better, Negative = Away is better.
    for col in ['roll_gf', 'roll_ga', 'roll_shots', 'roll_poss', 'roll_corners', 'cs_rate', 'power_score']:
        match_stats[f'{col}_diff'] = match_stats[f'{col}_home'] - match_stats[f'{col}_away']
    
    # 4. Target Encoding
    encoder = TargetEncoder(cols=['venue', 'referee'])
    match_stats[['venue_encoded', 'ref_encoded']] = encoder.fit_transform(match_stats[['venue', 'referee']], match_stats['target'])
    
    # Only keep the High-Signal Differential Features and Encoded categories
    final_cols = [
        'fixture_id', 'date', 'target', 'league_id', 'venue_encoded', 'ref_encoded',
        'roll_gf_diff', 'roll_ga_diff', 'roll_shots_diff', 'roll_poss_diff', 'roll_corners_diff', 
        'cs_rate_diff', 'power_score_diff'
    ]
    
    return match_stats[final_cols].fillna(0)

def process_matches_from_db():
    df = database.get_all_matches_df()
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_differential_features(df)
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info(f"SUCCESS: Differential Features complete. Shape: {df.shape}")
