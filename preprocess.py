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
    logging.info("🚀 Applying Strategy 7: Senior Feature Interactions & Efficiency Ratios...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Team-Level Processing
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'home_possession', 'home_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'away_possession', 'away_corners']].copy()
    h.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    a.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    group = team_stats.groupby('team_id')
    
    # Rolling stats for core metrics
    for col in ['gf', 'ga', 'sh', 'po', 'co']:
        team_stats[f'roll_{col}'] = group[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    # 2. INTERACTION ENGINEERING (The Logic Leap)
    # Offensive Efficiency: How many goals do you score per shot?
    team_stats['off_efficiency'] = (team_stats['roll_gf'] / (team_stats['roll_sh'] + 1)).replace([np.inf, -np.inf], 0)
    # Pressure Index: Do you control the ball AND win corners?
    team_stats['pressure_index'] = (team_stats['roll_po'] * team_stats['roll_co']) / 100
    # Defensive Resilience: How many shots can you survive before conceding?
    team_stats['def_resilience'] = (team_stats['roll_sh'] / (team_stats['roll_ga'] + 1)).replace([np.inf, -np.inf], 0)

    # 3. Merge Back
    match_stats = df.copy()
    for suffix in ['home', 'away']:
        tid = f'{suffix}_id'
        cols = ['fixture_id', 'team_id', 'roll_gf', 'roll_ga', 'roll_sh', 'roll_po', 'roll_co', 'off_efficiency', 'pressure_index', 'def_resilience']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', tid], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    # 4. DIFFERENTIALS (Gap Analysis)
    diff_cols = ['roll_gf', 'roll_ga', 'roll_sh', 'roll_po', 'roll_co', 'off_efficiency', 'pressure_index', 'def_resilience']
    for col in diff_cols:
        match_stats[f'{col}_diff'] = match_stats[f'{col}_home'] - match_stats[f'{col}_away']
    
    # 5. Target Encoding
    encoder = TargetEncoder(cols=['venue', 'referee'])
    match_stats[['venue_encoded', 'ref_encoded']] = encoder.fit_transform(match_stats[['venue', 'referee']], match_stats['target'])
    
    # Keep only High-Signal columns
    signal_features = [f'{c}_diff' for c in diff_cols] + ['venue_encoded', 'ref_encoded', 'league_id']
    final_cols = ['fixture_id', 'date', 'target'] + signal_features
    
    return match_stats[final_cols].fillna(0)

def process_matches_from_db():
    df = database.get_all_matches_df()
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_advanced_interactions(df)
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info(f"SUCCESS: Interaction Engineering complete. Features: {df.shape[1]}")
