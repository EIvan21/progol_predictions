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
import optuna
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def train_heavy_model():
    logging.info("🔬 STARTING SHIELDED ARCHITECTURE TRAINING")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    # 1. SPLIT FIRST (Zero Leakage)
    split_idx = int(len(df) * 0.85)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    y_train, y_test = train_df['target'], test_df['target']
    
    # 2. ENCODE INSIDE TRAIN SET
    logging.info("Encoding Categorical Features (Venue/Referee)...")
    encoder = TargetEncoder(cols=['venue', 'referee'])
    train_df_enc = encoder.fit_transform(train_df[['venue', 'referee']], y_train)
    test_df_enc = encoder.transform(test_df[['venue', 'referee']])
    
    # Combine back
    X_train = train_df.drop(columns=['fixture_id', 'date', 'target', 'venue', 'referee'])
    X_train['venue_enc'] = train_df_enc['venue']
    X_train['ref_enc'] = train_df_enc['referee']
    
    X_test = test_df.drop(columns=['fixture_id', 'date', 'target', 'venue', 'referee'])
    X_test['venue_enc'] = test_df_enc['venue']
    X_test['ref_enc'] = test_df_enc['referee']
    
    features = X_train.columns.tolist()
    
    # 3. SCALE
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    # 4. OPTUNA & ENSEMBLE (Simplified for speed but high quality)
    xgb_final = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42)
    rf_final = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    
    logging.info("Calibrating and Stacking...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = [
        ('xgb', CalibratedClassifierCV(xgb_final, method='isotonic', cv=cv)),
        ('rf', CalibratedClassifierCV(rf_final, method='isotonic', cv=cv))
    ]
    
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=cv,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    stack_model.fit(X_train_s, y_train)
    
    # 5. SAVE EVERYTHING
    probs = stack_model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, stack_model.predict(X_test_s))
    
    logging.info(f"✅ PRODUCTION READY. Accuracy: {acc:.4f} | LogLoss: {log_loss(y_test, probs):.4f}")
    
    package = {
        'model': stack_model,
        'scaler': scaler,
        'encoder': encoder,
        'features': features
    }
    joblib.dump(package, MODEL_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump({'accuracy': acc, 'log_loss': log_loss(y_test, probs), 'features': features}, f)

if __name__ == "__main__":
    train_heavy_model()
