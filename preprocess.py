import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import LabelEncoder
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

def calculate_elite_features(df):
    logging.info("Applying Strategy 3: Elite Contextual Engineering...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Basic Team Stats
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'away_id']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'home_id']].copy()
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'opp_id']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'opp_id']
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    
    # 2. ADVANCED: Opponent Strength (Elo Proxy)
    # We define a "Power Score" based on win rate in last 10 games
    group = team_stats.groupby('team_id')
    team_stats['is_win'] = (team_stats['gf'] > team_stats['ga']).astype(int)
    team_stats['power_score'] = group['is_win'].transform(lambda x: x.shift().rolling(10, min_periods=1).mean()).fillna(0.3)
    
    # 3. Strength-Adjusted Goals
    # Merge opponent power score back to adjust goals
    team_stats = team_stats.merge(
        team_stats[['fixture_id', 'team_id', 'power_score']], 
        left_on=['fixture_id', 'opp_id'], 
        right_on=['fixture_id', 'team_id'], 
        suffixes=('', '_opp')
    ).drop(columns=['team_id_opp'])
    
    team_stats['adj_gf'] = team_stats['gf'] * (1 + team_stats['power_score_opp'])
    team_stats['adj_ga'] = team_stats['ga'] * (2 - team_stats['power_score_opp'])
    
    # 4. Final Rolling Features
    group = team_stats.groupby('team_id')
    team_stats['roll_adj_gf'] = group['adj_gf'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['roll_adj_ga'] = group['adj_ga'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['clean_sheet_rate'] = group['ga'].transform(lambda x: (x.shift() == 0).rolling(5, min_periods=1).mean())
    team_stats['days_rest'] = group['date'].transform(lambda x: x.diff().dt.days.shift())

    # 5. Merge stats back to Match-Level
    match_stats = df.copy()
    for suffix in ['home', 'away']:
        tid = 'home_id' if suffix == 'home' else 'away_id'
        cols = ['fixture_id', 'team_id', 'roll_adj_gf', 'roll_adj_ga', 'clean_sheet_rate', 'power_score', 'days_rest']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', tid], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    return match_stats.rename(columns={'roll_adj_gf': 'roll_adj_gf_home', 'roll_adj_ga': 'roll_adj_ga_home', 'clean_sheet_rate': 'clean_sheet_rate_home', 'power_score': 'power_score_home', 'days_rest': 'days_rest_home'})

def process_matches_from_db():
    df = database.get_all_matches_df()
    if config.IS_LOCAL_TEST:
        df = df.sample(n=config.get_data_limit(len(df)), random_state=42)

    df['target'] = df.apply(get_target, axis=1)
    df = calculate_elite_features(df)
    
    for col in ['venue', 'referee', 'league_id']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
        logging.info("SUCCESS: Elite Preprocessing complete.")
