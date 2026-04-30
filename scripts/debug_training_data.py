import pandas as pd
import numpy as np
import os

DATA_PATH = 'data/processed/final_train_data.csv'

def debug_data():
    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Total rows in processed data: {len(df)}")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away', 'total_goals', 'result', 'year']
    features = [c for c in df.columns if c not in exclude]
    
    print("\n--- NAN COUNT PER FEATURE ---")
    nan_counts = df[features].isna().sum()
    print(nan_counts[nan_counts > 0])
    
    # Check if target has NaNs
    print(f"\nNaN in target: {df['target'].isna().sum()}")
    
    # Try the dropna logic
    df_clean = df.dropna(subset=features + ['target'])
    print(f"\nRows remaining after dropna: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("\nERROR: All rows were dropped!")
        for col in features:
            if df[col].isna().all():
                print(f"CRITICAL: Column '{col}' is entirely NaN.")

if __name__ == "__main__":
    debug_data()
