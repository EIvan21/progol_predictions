import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import config # Our fixed config module

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
    
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home']].copy()
    
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    team_stats['goal_diff'] = team_stats['goals_for'] - team_stats['goals_against']
    team_stats['points'] = team_stats.apply(lambda r: 3 if r['goal_diff'] > 0 else (1 if r['goal_diff'] == 0 else 0), axis=1)

    group = team_stats.groupby('team_id')
    
    # Define weighting logic
    def apply_weighting(x):
        if strategy == 0: # FLAT
            return x.shift().rolling(5, min_periods=1).mean()
        elif strategy == 1: # TEMPORAL
            return x.shift().ewm(span=span).mean() # In pandas, EWM on sorted series handles "recent" automatically
        elif strategy == 2: # ORDINAL
            return x.shift().ewm(alpha=0.4).mean() # Alpha based ordinal decay

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

def process_fixtures(raw_fixtures):
    if config.IS_LOCAL_TEST:
        limit = config.get_data_limit(len(raw_fixtures))
        logging.info(f"Local Test: Sampling {limit} matches.")
        raw_fixtures = raw_fixtures[:limit]

    rows = []
    for match in raw_fixtures:
        try:
            if match['goals']['home'] is None: continue
            rows.append({'fixture_id': match['fixture']['id'], 'league_id': match['league']['id'], 'date': match['fixture']['date'], 'venue': match['fixture']['venue']['name'] or "Unknown", 'referee': match['fixture']['referee'] or "Unknown", 'home_id': match['teams']['home']['id'], 'away_id': match['teams']['away']['id'], 'goals_home': match['goals']['home'], 'goals_away': match['goals']['away']})
        except: continue
    
    df = pd.DataFrame(rows)
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_stats_with_strategy(df)
    
    for col in ['venue', 'referee', 'league_id']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    all_fixtures = []
    if os.path.exists(config.RAW_DATA_DIR):
        files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.json')]
        for file in files:
            with open(os.path.join(config.RAW_DATA_DIR, file), 'r') as f:
                all_fixtures.extend(json.load(f).get('response', []))
        
        if all_fixtures:
            df = process_fixtures(all_fixtures)
            df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
            logging.info(f"SUCCESS: Processed {len(df)} matches.")
    else: logging.error("No raw data found.")
