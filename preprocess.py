import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import LabelEncoder
import config
import database # Our new database module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/preprocess.log"), logging.StreamHandler()]
)

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_stats_with_strategy(df, span=5):
    strategy = config.WEIGHT_STRATEGY
    logging.info(f"Applying Strategy {strategy}: {'Flat' if strategy == 0 else ('Temporal' if strategy == 1 else 'Ordinal')}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Same advanced stats logic as before
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home']].copy()
    
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    team_stats['goal_diff'] = team_stats['goals_for'] - team_stats['goals_against']
    team_stats['points'] = team_stats.apply(lambda r: 3 if r['goal_diff'] > 0 else (1 if r['goal_diff'] == 0 else 0), axis=1)

    group = team_stats.groupby('team_id')
    
    def apply_weighting(x):
        if strategy == 0: return x.shift().rolling(5, min_periods=1).mean()
        elif strategy == 1: return x.shift().ewm(span=span).mean()
        elif strategy == 2: return x.shift().ewm(alpha=0.4).mean()

    team_stats['ewm_goals_for'] = group['goals_for'].transform(apply_weighting)
    team_stats['ewm_goals_against'] = group['goals_against'].transform(apply_weighting)
    team_stats['ewm_goal_diff'] = group['goal_diff'].transform(apply_weighting)
    team_stats['ewm_form_points'] = group['points'].transform(apply_weighting)
    team_stats['days_rest'] = group['date'].transform(lambda x: x.diff().dt.days.shift())
    
    match_stats = df[['fixture_id', 'date', 'league_id', 'venue', 'referee', 'home_id', 'away_id', 'target']].copy()
    for suffix in ['home', 'away']:
        team_id_col = 'home_id' if suffix == 'home' else 'away_id'
        cols = ['fixture_id', 'team_id', 'ewm_goals_for', 'ewm_goals_against', 'ewm_goal_diff', 'ewm_form_points', 'days_rest']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', team_id_col], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    return match_stats.rename(columns={'ewm_goals_for': 'ewm_goals_for_home', 'ewm_goals_against': 'ewm_goals_against_home', 'ewm_goal_diff': 'ewm_goal_diff_home', 'ewm_form_points': 'ewm_form_points_home', 'days_rest': 'days_rest_home'})

def process_matches_from_db():
    # 1. Pull all finished matches from SQLite
    df = database.get_all_matches_df()
    logging.info(f"Loaded {len(df)} finished matches from Database.")

    if config.IS_LOCAL_TEST:
        limit = config.get_data_limit(len(df))
        logging.info(f"Local Test: Sampling {limit} matches.")
        df = df.sample(n=limit, random_state=42)

    # 2. Target Encoding
    df['target'] = df.apply(get_target, axis=1)
    
    # 3. Features
    df = calculate_stats_with_strategy(df)
    
    # 4. Encoding
    for col in ['venue', 'referee', 'league_id']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"SUCCESS: Preprocessed {len(df)} matches. Saved to {output_path}")
    else:
        logging.error("No matches to process from database.")
