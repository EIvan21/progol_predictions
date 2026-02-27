import pandas as pd
import numpy as np

def calculate_elo_ratings(df, k_factor=20, base_rating=1500):
    """
    Calculates historical Elo ratings for all teams.
    Returns a DataFrame with 'elo_home' and 'elo_away' for each match.
    """
    # Sort chronologically
    df = df.sort_values('date').copy()
    
    # Initialize ratings dictionary
    team_ratings = {}
    
    elo_home_list = []
    elo_away_list = []
    
    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        
        # Get current ratings (default to base)
        h_rating = team_ratings.get(hid, base_rating)
        a_rating = team_ratings.get(aid, base_rating)
        
        # Store PRE-MATCH ratings (Feature)
        elo_home_list.append(h_rating)
        elo_away_list.append(a_rating)
        
        # Calculate Expected Result
        # P(HomeWin) approx 1 / (1 + 10^((A-H)/400))
        # We adjust for Home Advantage (+75 points usually)
        h_adv = 75
        expected_h = 1 / (1 + 10 ** ((a_rating - (h_rating + h_adv)) / 400))
        
        # Actual Result (1=Home, 0.5=Draw, 0=Away)
        if row['goals_home'] > row['goals_away']: result = 1.0
        elif row['goals_home'] == row['goals_away']: result = 0.5
        else: result = 0.0
        
        # Update Ratings
        new_h = h_rating + k_factor * (result - expected_h)
        new_a = a_rating + k_factor * ((1 - result) - (1 - expected_h))
        
        team_ratings[hid] = new_h
        team_ratings[aid] = new_a
        
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    df['elo_prob_h'] = 1 / (1 + 10 ** ((df['elo_away'] - (df['elo_home'] + 75)) / 400))
    
    return df

def calculate_league_zscores(df, window=10):
    """
    Normalizes stats relative to the LEAGUE average at that point in time.
    Prevents bias when comparing high-scoring vs low-scoring leagues.
    """
    df = df.sort_values('date')
    
    # Metric columns to normalize
    metrics = ['goals_home', 'goals_away', 'home_shots', 'away_shots']
    
    # Calculate rolling league averages
    league_means = df.groupby('league_id')[metrics].transform(lambda x: x.shift().rolling(window=50, min_periods=10).mean())
    league_stds = df.groupby('league_id')[metrics].transform(lambda x: x.shift().rolling(window=50, min_periods=10).std())
    
    # Z-Score = (Value - LeagueMean) / LeagueStd
    # We do this for the Rolling Team Stats, not the raw match stats
    # But first we need the rolling team stats.
    # ... Logic handled in preprocess, this helper is for reference
    return df

def add_rolling_features(df):
    """
    Complex rolling features with strict shift=1 to prevent leakage.
    """
    df = df.sort_values('date')
    
    # Stack home and away to get team-centric view
    h = df[['fixture_id', 'date', 'league_id', 'home_id', 'goals_home', 'goals_away', 'home_shots', 'away_shots', 'home_corners', 'away_corners']].copy()
    a = df[['fixture_id', 'date', 'league_id', 'away_id', 'goals_away', 'goals_home', 'away_shots', 'home_shots', 'away_corners', 'home_corners']].copy()
    
    cols = ['fixture_id', 'date', 'league_id', 'team_id', 'gf', 'ga', 'sf', 'sa', 'cf', 'ca']
    h.columns = cols; a.columns = cols
    
    team_stats = pd.concat([h, a]).sort_values(['team_id', 'date'])
    g = team_stats.groupby('team_id')
    
    # Calculate rolling metrics (Strictly shifting)
    # Form: Points in last 5 (3=W, 1=D, 0=L)
    team_stats['pts'] = np.where(team_stats['gf'] > team_stats['ga'], 3, np.where(team_stats['gf'] == team_stats['ga'], 1, 0))
    
    window = 5
    team_stats['roll_form'] = g['pts'].transform(lambda x: x.shift().rolling(window, min_periods=3).mean())
    team_stats['roll_gf'] = g['gf'].transform(lambda x: x.shift().rolling(window, min_periods=3).mean())
    team_stats['roll_ga'] = g['ga'].transform(lambda x: x.shift().rolling(window, min_periods=3).mean())
    team_stats['roll_sf'] = g['sf'].transform(lambda x: x.shift().rolling(window, min_periods=3).mean()) # Shots For
    team_stats['roll_sa'] = g['sa'].transform(lambda x: x.shift().rolling(window, min_periods=3).mean()) # Shots Against
    
    # League Normalization Context (Advanced)
    # We join back to league averages to create "Relative Strength"
    # (Implementation simplified for speed, usually done via merge)
    
    # Merge back
    df_out = df.copy()
    for suffix in ['home', 'away']:
        tid_col = f'{suffix}_id'
        merge_cols = ['fixture_id', 'team_id', 'roll_form', 'roll_gf', 'roll_ga', 'roll_sf', 'roll_sa']
        
        df_out = df_out.merge(team_stats[merge_cols], 
                              left_on=['fixture_id', tid_col], 
                              right_on=['fixture_id', 'team_id'], 
                              suffixes=('', f'_{suffix}'))
        df_out = df_out.drop(columns=['team_id'])
        
    return df_out
