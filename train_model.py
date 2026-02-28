import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_A_PATH = 'models/binary_home_detector.pkl' # Brain 1: Home vs Not-Home
MODEL_B_PATH = 'models/draw_away_separator.pkl' # Brain 2: Draw vs Away
SCALER_PATH = 'models/scaler.pkl'

def train_cascading_models():
    logging.info("🔬 STARTING HIERARCHICAL (CASCADING) TRAINING")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target', 'venue', 'referee']
    features = [c for c in df.columns if c not in exclude]
    
    # 1. Prepare Data
    df = df.dropna(subset=['target'])
    X = df[features].fillna(0)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # --- MODEL A: HOME vs NOT-HOME ---
    logging.info("Training Model A (Home Win Detector)...")
    y_binary = (y == 0).astype(int) # 1 if Home, 0 otherwise
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_scaled, y_binary, test_size=0.15, stratify=y_binary, random_state=42)
    
    model_a = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05)
    model_a.fit(X_train_a, y_train_a)
    acc_a = accuracy_score(y_test_a, model_a.predict(X_test_a))
    logging.info(f"Model A Accuracy: {acc_a:.4f}")

    # --- MODEL B: DRAW vs AWAY ---
    # We only train on rows where target was NOT Home (target != 0)
    logging.info("Training Model B (Draw vs Away Specialist)...")
    df_non_home = df[df['target'] != 0].copy()
    X_nh = X_scaled.loc[df_non_home.index]
    y_nh = df_non_home['target'] # Will be 1 (Draw) or 2 (Away)
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_nh, y_nh, test_size=0.15, stratify=y_nh, random_state=42)
    
    model_b = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced')
    model_b.fit(X_train_b, y_train_b)
    acc_b = accuracy_score(y_test_b, model_b.predict(X_test_b))
    logging.info(f"Model B Accuracy: {acc_b:.4f}")

    # Save Everything
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_a, MODEL_A_PATH)
    joblib.dump(model_b, MODEL_B_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    with open('models/metrics.json', 'w') as f:
        json.dump({'model_a_acc': acc_a, 'model_b_acc': acc_b, 'features': features}, f)
    
    logging.info("✅ Cascading Brains Saved Successfully.")

if __name__ == "__main__":
    train_cascading_models()
