import pandas as pd
import numpy as np

def calculate_elo_ratings(df, k_factor=20, base_rating=1500):
    df = df.sort_values('date').copy()
    team_ratings = {}
    elo_home_list, elo_away_list = [], []
    
    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        h_rating = team_ratings.get(hid, base_rating)
        a_rating = team_ratings.get(aid, base_rating)
        elo_home_list.append(h_rating)
        elo_away_list.append(a_rating)
        
        h_adv = 75
        expected_h = 1 / (1 + 10 ** ((a_rating - (h_rating + h_adv)) / 400))
        result = 1.0 if row['goals_home'] > row['goals_away'] else (0.5 if row['goals_home'] == row['goals_away'] else 0.0)
        
        new_h = h_rating + k_factor * (result - expected_h)
        new_a = a_rating + k_factor * ((1 - result) - (1 - expected_h))
        team_ratings[hid], team_ratings[aid] = new_h, new_a
        
    df['elo_home'], df['elo_away'] = elo_home_list, elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    df['elo_prob_h'] = 1 / (1 + 10 ** ((df['elo_away'] - (df['elo_home'] + 75)) / 400))
    return df

def add_rolling_features(df):
    df = df.sort_values('date')
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'away_shots']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'home_shots']].copy()
    cols = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sf', 'sa']
    h.columns = cols; a.columns = cols
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    g = team_stats.groupby('team_id')
    team_stats['pts'] = np.where(team_stats['gf'] > team_stats['ga'], 3, np.where(team_stats['gf'] == team_stats['ga'], 1, 0))
    
    metrics = ['form', 'gf', 'ga', 'sf', 'sa']
    team_stats['roll_form'] = g['pts'].transform(lambda x: x.shift().rolling(5, min_periods=3).mean())
    for m in ['gf', 'ga', 'sf', 'sa']:
        team_stats[f'roll_{m}'] = g[m].transform(lambda x: x.shift().rolling(5, min_periods=3).mean())
    
    # MERGE LOGIC (Explicit renaming to avoid KeyError)
    df_out = df.copy()
    feature_cols = ['roll_form', 'roll_gf', 'roll_ga', 'roll_sf', 'roll_sa']
    
    # 1. Merge Home
    home_stats = team_stats[['fixture_id', 'team_id'] + feature_cols].copy()
    home_stats.columns = ['fixture_id', 'home_id'] + [f'{c}_home' for c in feature_cols]
    df_out = df_out.merge(home_stats, on=['fixture_id', 'home_id'], how='left')
    
    # 2. Merge Away
    away_stats = team_stats[['fixture_id', 'team_id'] + feature_cols].copy()
    away_stats.columns = ['fixture_id', 'away_id'] + [f'{c}_away' for c in feature_cols]
    df_out = df_out.merge(away_stats, on=['fixture_id', 'away_id'], how='left')
    
    return df_out
