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

def calculate_advanced_interactions(df):
    logging.info("🚀 Applying Strategy 7: Senior Feature Interactions...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'home_possession', 'home_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'away_possession', 'away_corners']].copy()
    h.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    a.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    group = team_stats.groupby('team_id')
    
    cols_to_roll = ['gf', 'ga', 'sh', 'po', 'co']
    for col in cols_to_roll:
        team_stats[f'roll_{col}'] = group[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    team_stats['off_efficiency'] = (team_stats['roll_gf'] / (team_stats['roll_sh'] + 1))
    team_stats['pressure_index'] = (team_stats['roll_po'] * team_stats['roll_co']) / 100
    team_stats['def_resilience'] = (team_stats['roll_sh'] / (team_stats['roll_ga'] + 1))

    match_stats = df.copy()
    feature_cols = ['roll_gf', 'roll_ga', 'roll_sh', 'roll_po', 'roll_co', 'off_efficiency', 'pressure_index', 'def_resilience']
    
    home_features = team_stats[['fixture_id', 'team_id'] + feature_cols].copy()
    home_features.columns = ['fixture_id', 'home_id'] + [f'{c}_home' for c in feature_cols]
    match_stats = match_stats.merge(home_features, on=['fixture_id', 'home_id'], how='left')
    
    away_features = team_stats[['fixture_id', 'team_id'] + feature_cols].copy()
    away_features.columns = ['fixture_id', 'away_id'] + [f'{c}_away' for c in feature_cols]
    match_stats = match_stats.merge(away_features, on=['fixture_id', 'away_id'], how='left')

    for col in feature_cols:
        match_stats[f'{col}_diff'] = match_stats[f'{col}_home'] - match_stats[f'{col}_away']
    
    encoder = TargetEncoder(cols=['venue', 'referee'])
    match_stats[['venue_encoded', 'ref_encoded']] = encoder.fit_transform(match_stats[['venue', 'referee']], match_stats['target'])
    
    diff_features = [f'{c}_diff' for c in feature_cols]
    final_cols = ['fixture_id', 'date', 'target', 'league_id', 'venue_encoded', 'ref_encoded'] + diff_features
    return match_stats[final_cols].fillna(0)

def process_matches_from_db():
    df = database.get_all_matches_df()
    if df.empty:
        logging.error("DATABASE IS EMPTY. Run fetch_data.py first.")
        return df
        
    if config.IS_LOCAL_TEST:
        limit = config.get_data_limit(len(df))
        logging.info(f"LOCAL TEST MODE: Sampling {limit} matches.")
        df = df.sample(n=limit, random_state=42)

    df['target'] = df.apply(get_target, axis=1)
    df = calculate_advanced_interactions(df)
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
        logging.info("SUCCESS: Preprocessing complete.")
