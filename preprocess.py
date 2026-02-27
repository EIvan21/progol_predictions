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

def calculate_full_features(df):
    logging.info("🚀 Calculating Full Feature Set (Stats + Context + Power Score)...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Team-Level Processing
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'home_possession', 'home_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'away_possession', 'away_corners']].copy()
    h.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'shots', 'poss', 'corners']
    a.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'shots', 'poss', 'corners']
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    group = team_stats.groupby('team_id')
    
    # POWER SCORE (Elo Proxy)
    team_stats['is_win'] = (team_stats['gf'] > team_stats['ga']).astype(int)
    team_stats['power_score'] = group['is_win'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean()).fillna(0.3)
    
    # ROLLING STATS (Window=5)
    for col in ['gf', 'ga', 'shots', 'poss', 'corners']:
        team_stats[f'roll_{col}'] = group[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(0)
    
    team_stats['cs_rate'] = group['ga'].transform(lambda x: (x.shift() == 0).rolling(5, min_periods=1).mean()).fillna(0)
    team_stats['days_rest'] = group['date'].transform(lambda x: x.diff().dt.days.shift())

    # 2. League Home Advantage Factor
    ha_map = df.groupby('league_id').apply(lambda x: (x['goals_home'] > x['goals_away']).mean(), include_groups=False).to_dict()
    df['league_ha_factor'] = df['league_id'].map(ha_map)

    # 3. Merge Back
    match_stats = df.copy()
    for suffix in ['home', 'away']:
        tid = f'{suffix}_id'
        cols = ['fixture_id', 'team_id', 'roll_gf', 'roll_ga', 'roll_shots', 'roll_poss', 'roll_corners', 'cs_rate', 'power_score', 'days_rest']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', tid], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    # 4. Target Encoding
    logging.info("Applying Target Encoding to Venue and Referee...")
    encoder = TargetEncoder(cols=['venue', 'referee'])
    match_stats[['venue_encoded', 'ref_encoded']] = encoder.fit_transform(match_stats[['venue', 'referee']], match_stats['target'])
    
    return match_stats.rename(columns={
        'roll_gf': 'roll_gf_home', 'roll_ga': 'roll_ga_home', 'roll_shots': 'roll_shots_home', 
        'roll_poss': 'roll_poss_home', 'roll_corners': 'roll_corners_home', 'cs_rate': 'cs_rate_home',
        'power_score': 'power_score_home', 'days_rest': 'days_rest_home'
    })

def process_matches_from_db():
    df = database.get_all_matches_df()
    df = df.dropna(subset=['home_shots']) # Only use matches with full stats
    logging.info(f"Processing {len(df)} matches with full statistical depth.")
    
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_full_features(df)
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
        logging.info("SUCCESS: Full Features Preprocessing complete.")
