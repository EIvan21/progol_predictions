import pandas as pd
import numpy as np
import os
import logging
import config
import database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_clean_features(df):
    logging.info("🚀 Calculating Differential Features (No Leakage)...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Team-Level Processing
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'home_possession', 'home_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'away_possession', 'away_corners']].copy()
    h.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    a.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sh', 'po', 'co']
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    group = team_stats.groupby('team_id')
    
    # Strictly shifted rolling stats
    for col in ['gf', 'ga', 'sh', 'po', 'co']:
        team_stats[f'roll_{col}'] = group[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    # Efficiency Metrics
    team_stats['off_eff'] = (team_stats['roll_gf'] / (team_stats['roll_sh'] + 1))
    team_stats['press_idx'] = (team_stats['roll_po'] * team_stats['roll_co']) / 100
    
    # 2. Merge Back
    match_stats = df.copy()
    f_cols = ['roll_gf', 'roll_ga', 'roll_sh', 'roll_po', 'roll_co', 'off_eff', 'press_idx']
    
    h_f = team_stats[['fixture_id', 'team_id'] + f_cols].copy()
    h_f.columns = ['fixture_id', 'home_id'] + [f'{c}_h' for c in f_cols]
    match_stats = match_stats.merge(h_f, on=['fixture_id', 'home_id'], how='left')
    
    a_f = team_stats[['fixture_id', 'team_id'] + f_cols].copy()
    a_f.columns = ['fixture_id', 'away_id'] + [f'{c}_a' for c in f_cols]
    match_stats = match_stats.merge(a_f, on=['fixture_id', 'away_id'], how='left')

    # 3. Differentials
    for col in f_cols:
        match_stats[f'{col}_diff'] = match_stats[f'{col}_h'] - match_stats[f'{col}_a']
    
    # 4. Filter and return
    target_cols = [f'{c}_diff' for c in f_cols] + ['league_id', 'venue', 'referee']
    final_df = match_stats[['fixture_id', 'date', 'target'] + target_cols]
    return final_df.fillna(0)

def process_matches_from_db():
    df = database.get_all_matches_df()
    df['target'] = df.apply(get_target, axis=1)
    return calculate_clean_features(df)

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info(f"SUCCESS: Preprocessing complete. Matches: {len(df)}")
