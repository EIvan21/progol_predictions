import pandas as pd
import numpy as np

def calculate_elo_ratings(df, k_factor=20, base_rating=1500):
    """
    Calculates Elo ratings and returns the dataframe plus the history dict.
    """
    df = df.sort_values('date').copy()
    team_ratings = {} # Current rating
    rating_history = [] # To store (date, team_id, rating) for SoS lookup
    
    elo_h_col, elo_a_col = [], []
    
    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        date = row['date']
        
        # Get current ratings (before match)
        h_rating = team_ratings.get(hid, base_rating)
        a_rating = team_ratings.get(aid, base_rating)
        
        elo_h_col.append(h_rating)
        elo_a_col.append(a_rating)
        
        # Store for SoS lookup later
        rating_history.append({'date': date, 'team_id': hid, 'rating': h_rating})
        rating_history.append({'date': date, 'team_id': aid, 'rating': a_rating})
        
        # Calculate new ratings (after match)
        h_adv = 75 # Home field advantage
        
        # Win Probability
        dr = a_rating - (h_rating + h_adv)
        expected_h = 1 / (1 + 10 ** (dr / 400))
        
        # Result (1=Win, 0.5=Draw, 0=Loss)
        if row['goals_home'] > row['goals_away']: result = 1.0
        elif row['goals_home'] == row['goals_away']: result = 0.5
        else: result = 0.0
        
        new_h = h_rating + k_factor * (result - expected_h)
        new_a = a_rating + k_factor * ((1 - result) - (1 - expected_h))
        
        team_ratings[hid] = new_h
        team_ratings[aid] = new_a
        
    df['elo_home'] = elo_h_col
    df['elo_away'] = elo_a_col
    
    # Create a quick lookup dataframe for SoS
    history_df = pd.DataFrame(rating_history).drop_duplicates(subset=['date', 'team_id'], keep='last')
    
    return df, history_df

def add_rolling_features(df, history_df=None):
    """
    Adds EWMA (Exponential Moving Average) and SoS (Strength of Schedule) features.
    """
    df = df.sort_values('date')
    
    # Prepare long-format stats
    h = df[['fixture_id', 'date', 'home_id', 'away_id', 'goals_home', 'goals_away', 'home_shots', 'away_shots']].copy()
    h.columns = ['fixture_id', 'date', 'team_id', 'opponent_id', 'gf', 'ga', 'sf', 'sa']
    
    a = df[['fixture_id', 'date', 'away_id', 'home_id', 'goals_away', 'goals_home', 'away_shots', 'home_shots']].copy()
    a.columns = ['fixture_id', 'date', 'team_id', 'opponent_id', 'gf', 'ga', 'sf', 'sa']
    
    stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    
    # 1. Add Opponent Elo (SoS)
    if history_df is not None:
        stats = stats.merge(history_df, left_on=['date', 'opponent_id'], right_on=['date', 'team_id'], suffixes=('', '_opp'), how='left')
        stats['opp_rating'] = stats['rating'].fillna(1500)
    else:
        stats['opp_rating'] = 1500

    # 2. Calculate Weighted Performance (SoS Adjustment)
    # A goal against a 1800 team is worth more than against a 1200 team
    # Adjustment Factor = OpponentElo / AverageElo(1500)
    stats['sos_factor'] = stats['opp_rating'] / 1500.0
    
    stats['w_gf'] = stats['gf'] * stats['sos_factor']
    stats['w_sf'] = stats['sf'] * stats['sos_factor']
    
    # 3. Calculate EWMA (Span=5 matches ~ alpha=0.33)
    # This weights recent games much higher than older games
    cols_to_avg = ['gf', 'ga', 'sf', 'sa', 'w_gf', 'w_sf']
    
    grouped = stats.groupby('team_id')[cols_to_avg]
    
    # We shift(1) because we want stats BEFORE the match starts
    ewma = grouped.apply(lambda x: x.shift().ewm(span=5, min_periods=3).mean())
    
    # Join back
    stats = stats.join(ewma, rsuffix='_ewma')
    
    # Extract final features for merge
    final_cols = ['fixture_id', 'team_id', 'gf_ewma', 'ga_ewma', 'sf_ewma', 'sa_ewma', 'w_gf_ewma', 'w_sf_ewma']
    
    # Merge Home
    df = df.merge(stats[final_cols], left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={c: f'home_{c}' for c in final_cols if c not in ['fixture_id', 'team_id']})
    df = df.drop(columns=['team_id'])
    
    # Merge Away
    df = df.merge(stats[final_cols], left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={c: f'away_{c}' for c in final_cols if c not in ['fixture_id', 'team_id']})
    df = df.drop(columns=['team_id'])
    
    return df
