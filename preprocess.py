import pandas as pd
import numpy as np
import os
import logging
from category_encoders import TargetEncoder
import config
import database
import features # The new signal engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_target(row):
    if row['goals_home'] > row['goals_away']: return 0
    elif row['goals_home'] == row['goals_away']: return 1
    else: return 2

def process_matches_from_db():
    df = database.get_all_matches_df()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    logging.info(f"Loaded {len(df)} matches. Calculating Advanced Signals...")
    
    # 1. ELO RATINGS (The strongest single predictor)
    df = features.calculate_elo_ratings(df)
    
    # 2. ROLLING FORM & STATS (Strict Shift=1)
    df = features.add_rolling_features(df)
    
    # 3. DIFFERENTIALS (Home - Away)
    cols = ['roll_form', 'roll_gf', 'roll_ga', 'roll_sf', 'roll_sa']
    for c in cols:
        df[f'{c}_diff'] = df[f'{c}_home'] - df[f'{c}_away']
        
    # 4. TARGET ENCODING (Venue & Referee)
    # We use LOO (Leave-One-Out) or K-Fold encoding in training to prevent leakage
    # For now, we use TargetEncoder from category_encoders which handles some smoothing
    df['target'] = df.apply(get_target, axis=1)
    
    encoder = TargetEncoder(cols=['venue', 'referee'])
    df[['venue_enc', 'ref_enc']] = encoder.fit_transform(df[['venue', 'referee']], df['target'])
    
    # Clean
    final_cols = [
        'fixture_id', 'date', 'target', 'league_id', 
        'elo_prob_h', 'elo_diff', 
        'roll_form_diff', 'roll_gf_diff', 'roll_ga_diff', 'roll_sf_diff', 'roll_sa_diff',
        'venue_enc', 'ref_enc'
    ]
    
    # Filter valid rows (must have history)
    df_clean = df.dropna(subset=['roll_form_diff'])
    logging.info(f"Cleaned Dataset: {len(df_clean)} rows ready for Scientific Training.")
    
    return df_clean

if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "final_train_data.csv"), index=False)
    logging.info("SUCCESS: Feature Engineering Complete.")
