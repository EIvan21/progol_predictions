import pandas as pd
import numpy as np
import os
import logging
from category_encoders import TargetEncoder
import config
import database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/preprocess.log"), logging.StreamHandler()]
)

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_hyper_features(df):
    logging.info("Applying Strategy 5: Hyper-Ensemble Engineering...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'away_id']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'home_id']].copy()
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'opp_id']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'opp_id']
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    
    group = team_stats.groupby('team_id')
    team_stats['is_win'] = (team_stats['gf'] > team_stats['ga']).astype(int)
    team_stats['power_score'] = group['is_win'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean()).fillna(0.3)
    
    team_stats = team_stats.merge(team_stats[['fixture_id', 'team_id', 'power_score']], left_on=['fixture_id', 'opp_id'], right_on=['fixture_id', 'team_id'], suffixes=('', '_opp')).drop(columns=['team_id_opp'])
    team_stats['adj_gf'] = team_stats['gf'] * (1 + team_stats['power_score_opp'])
    team_stats['adj_ga'] = team_stats['ga'] * (2 - team_stats['power_score_opp'])
    
    group = team_stats.groupby('team_id')
    team_stats['roll_gf'] = group['adj_gf'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['roll_ga'] = group['adj_ga'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['cs_rate'] = group['ga'].transform(lambda x: (x.shift() == 0).rolling(5, min_periods=1).mean())
    
    # Fixed Groupby Warning
    league_ha = df.groupby('league_id').apply(lambda x: (x['goals_home'] > x['goals_away']).mean(), include_groups=False).to_dict()
    df['league_ha_factor'] = df['league_id'].map(league_ha)

    match_stats = df.copy()
    for suffix in ['home', 'away']:
        tid = 'home_id' if suffix == 'home' else 'away_id'
        cols = ['fixture_id', 'team_id', 'roll_gf', 'roll_ga', 'cs_rate', 'power_score']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', tid], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    logging.info("Applying Target Encoding to Venue and Referee...")
    # Robust Encoding fit
    encoder = TargetEncoder(cols=['venue', 'referee'])
    match_stats[['venue_encoded', 'ref_encoded']] = encoder.fit_transform(match_stats[['venue', 'referee']], match_stats['target'])
    
    return match_stats.rename(columns={'roll_gf': 'roll_gf_home', 'roll_ga': 'roll_ga_home', 'cs_rate': 'cs_rate_home', 'power_score': 'power_score_home'})

def process_matches_from_db():
    df = database.get_all_matches_df()
    if config.IS_LOCAL_TEST: df = df.sample(n=config.get_data_limit(len(df)), random_state=42)
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_hyper_features(df)
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
        logging.info("SUCCESS: Hyper-Ensemble Preprocessing complete.")
