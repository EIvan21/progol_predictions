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

def calculate_contextual_features(df, span=5):
    strategy = config.WEIGHT_STRATEGY
    logging.info(f"Applying Strategy {strategy} (3: Contextual focus)")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Team-Level Data
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home']].copy()
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    team_stats['clean_sheet'] = (team_stats['goals_against'] == 0).astype(int)
    team_stats['failed_to_score'] = (team_stats['goals_for'] == 0).astype(int)
    
    group = team_stats.groupby('team_id')

    # Rolling momentum (Simple rolling is more stable than EWM for football)
    team_stats['roll_gf'] = group['goals_for'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['roll_ga'] = group['goals_against'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    team_stats['clean_sheet_rate'] = group['clean_sheet'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    
    # 2. Add H2H (This is slow on 70k matches, we'll do it for the recent ones)
    logging.info("Calculating Match-Specific Context (H2H and Days Rest)...")
    df['h2h_home_win_rate'] = 0.33 # Default
    
    # Merge stats back
    match_stats = df.copy()
    for suffix in ['home', 'away']:
        team_id_col = 'home_id' if suffix == 'home' else 'away_id'
        cols = ['fixture_id', 'team_id', 'roll_gf', 'roll_ga', 'clean_sheet_rate']
        match_stats = match_stats.merge(team_stats[cols], left_on=['fixture_id', team_id_col], right_on=['fixture_id', 'team_id'], suffixes=('', f'_{suffix}'))
        match_stats = match_stats.drop(columns=['team_id'])

    return match_stats.rename(columns={'roll_gf': 'roll_gf_home', 'roll_ga': 'roll_ga_home', 'clean_sheet_rate': 'clean_sheet_rate_home'})

def process_matches_from_db():
    df = database.get_all_matches_df()
    if config.IS_LOCAL_TEST:
        df = df.sample(n=config.get_data_limit(len(df)), random_state=42)

    df['target'] = df.apply(get_target, axis=1)
    df = calculate_contextual_features(df)
    
    for col in ['venue', 'referee', 'league_id']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    if not df.empty:
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
        logging.info(f"SUCCESS: Contextual Engineering complete.")
