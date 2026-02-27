import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/preprocess.log"), logging.StreamHandler()]
)

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def calculate_advanced_stats(df, span=5):
    """Calculates Exponentially Weighted Moving Averages and Form Streaks."""
    logging.info(f"Calculating Advanced Stats (EWMA span={span})...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create Team-Level Data
    home = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away']].copy()
    away = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home']].copy()
    
    home.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    away.columns = ['fixture_id', 'date', 'league_id', 'team_id', 'goals_for', 'goals_against']
    
    team_stats = pd.concat([home, away]).sort_values(['team_id', 'date'])
    
    # Feature 1: Goal Difference per match
    team_stats['goal_diff'] = team_stats['goals_for'] - team_stats['goals_against']
    
    # Feature 2: Result (3 for win, 1 for draw, 0 for loss)
    def get_points(row):
        if row['goal_diff'] > 0: return 3
        if row['goal_diff'] == 0: return 1
        return 0
    team_stats['points'] = team_stats.apply(get_points, axis=1)

    group = team_stats.groupby('team_id')
    
    # FEATURE ENGINEERING: Exponentially Weighted Moving Averages (EWM)
    # This gives significantly more weight to recent matches!
    team_stats['ewm_goals_for'] = group['goals_for'].transform(lambda x: x.shift().ewm(span=span).mean())
    team_stats['ewm_goals_against'] = group['goals_against'].transform(lambda x: x.shift().ewm(span=span).mean())
    team_stats['ewm_goal_diff'] = group['goal_diff'].transform(lambda x: x.shift().ewm(span=span).mean())
    team_stats['ewm_form_points'] = group['points'].transform(lambda x: x.shift().ewm(span=span).mean())
    
    # Days Rest
    team_stats['days_rest'] = group['date'].transform(lambda x: x.diff().dt.days.shift())
    
    # Re-merge to Match-Level
    match_stats = df[['fixture_id', 'date', 'league_id', 'venue', 'referee', 'home_id', 'away_id', 'target']].copy()
    
    for suffix in ['home', 'away']:
        team_id_col = 'home_id' if suffix == 'home' else 'away_id'
        cols_to_merge = ['fixture_id', 'team_id', 'ewm_goals_for', 'ewm_goals_against', 'ewm_goal_diff', 'ewm_form_points', 'days_rest']
        
        match_stats = match_stats.merge(
            team_stats[cols_to_merge],
            left_on=['fixture_id', team_id_col], 
            right_on=['fixture_id', 'team_id'], 
            suffixes=('', f'_{suffix}')
        )
        match_stats = match_stats.drop(columns=['team_id'])

    # Clean up column names from the first merge
    match_stats = match_stats.rename(columns={
        'ewm_goals_for': 'ewm_goals_for_home',
        'ewm_goals_against': 'ewm_goals_against_home',
        'ewm_goal_diff': 'ewm_goal_diff_home',
        'ewm_form_points': 'ewm_form_points_home',
        'days_rest': 'days_rest_home'
    })
    
    return match_stats

def process_fixtures(raw_fixtures):
    rows = []
    for match in raw_fixtures:
        try:
            if match['goals']['home'] is None: continue
            rows.append({
                'fixture_id': match['fixture']['id'],
                'league_id': match['league']['id'],
                'date': match['fixture']['date'],
                'venue': match['fixture']['venue']['name'] or "Unknown",
                'referee': match['fixture']['referee'] or "Unknown",
                'home_id': match['teams']['home']['id'],
                'away_id': match['teams']['away']['id'],
                'goals_home': match['goals']['home'],
                'goals_away': match['goals']['away']
            })
        except: continue
        
    df = pd.DataFrame(rows)
    df['target'] = df.apply(get_target, axis=1)
    df = calculate_advanced_stats(df)
    
    for col in ['venue', 'referee', 'league_id']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    return df

if __name__ == "__main__":
    RAW_DIR, PROCESSED_DIR = "data/raw/", "data/processed/"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_fixtures = []
    
    if os.path.exists(RAW_DIR):
        files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        for file in files:
            with open(os.path.join(RAW_DIR, file), 'r') as f:
                data = json.load(f)
                all_fixtures.extend(data.get('response', []))
        
        if all_fixtures:
            df = process_fixtures(all_fixtures)
            df.to_csv(os.path.join(PROCESSED_DIR, "final_train_data.csv"), index=False)
            logging.info(f"SUCCESS: Processed {len(df)} matches with EWMA features.")
    else:
        logging.error("No raw data found.")
