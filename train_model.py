import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
PRIMARY_MODEL_PATH = 'models/calibrated_ensemble.pkl'
UNDERDOG_MODEL_PATH = 'models/underdog_specialist.pkl'
METRICS_PATH = 'models/metrics.json'

def train_heavy_model():
    logging.info("🔬 STARTING MULTI-PERSPECTIVE TRAINING")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    split_idx = int(len(df) * 0.85)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    # 1. PREPARE ENCODER & SCALER
    encoder = TargetEncoder(cols=['venue', 'referee'])
    train_df_enc = encoder.fit_transform(train_df[['venue', 'referee']], train_df['target'])
    
    def get_X_y(base_df, enc_df):
        X = base_df.drop(columns=['fixture_id', 'date', 'target', 'venue', 'referee'])
        X['venue_enc'] = enc_df['venue']
        X['ref_enc'] = enc_df['referee']
        return X, base_df['target']

    X_train, y_train = get_X_y(train_df, train_df_enc)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    
    # --- PERSPECTIVE 1: STANDARD ENSEMBLE ---
    logging.info("Training Primary Ensemble (L/E/V)...")
    base_models = [
        ('xgb', CalibratedClassifierCV(xgb.XGBClassifier(n_estimators=300), method='isotonic')),
        ('rf', CalibratedClassifierCV(RandomForestClassifier(n_estimators=300), method='isotonic'))
    ]
    primary_stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3, stack_method='predict_proba')
    primary_stack.fit(X_train_s, y_train)

    # --- PERSPECTIVE 2: UNDERDOG SPECIALIST (E vs V) ---
    logging.info("Training Underdog Specialist (E vs V)...")
    nh_idx = y_train != 0
    X_train_nh = X_train_s[nh_idx]
    y_train_nh = y_train[nh_idx]
    
    underdog_stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3, stack_method='predict_proba')
    underdog_stack.fit(X_train_nh, y_train_nh)

    # 3. SAVE EVERYTHING
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': primary_stack, 'scaler': scaler, 'encoder': encoder, 'features': X_train.columns.tolist()}, PRIMARY_MODEL_PATH)
    joblib.dump({'model': underdog_stack}, UNDERDOG_MODEL_PATH)
    
    logging.info("✅ Both Brains Saved Successfully.")

if __name__ == "__main__":
    train_heavy_model()
