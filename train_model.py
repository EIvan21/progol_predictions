import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def progol_metric(y_true, y_probs):
    """
    Calculates Expected Correct Picks per 14-game ticket.
    Max score = 14. Random guessing = 4.66.
    """
    # Sum of probabilities for the correct class
    # y_true is (N,), y_probs is (N, 3)
    # We need to pick the prob corresponding to the true class
    n = len(y_true)
    correct_prob_sum = 0
    for i in range(n):
        true_class = int(y_true.iloc[i])
        correct_prob_sum += y_probs[i, true_class]
    
    avg_prob = correct_prob_sum / n
    expected_hits = avg_prob * 14
    return expected_hits

def train_scientific_model():
    logging.info("🔬 STARTING SCIENTIFIC CALIBRATION PIPELINE")
    
    df = pd.read_csv(DATA_PATH)
    
    # Time-Based Split (Walk-Forward)
    # We sort by date to ensure we train on past, test on future
    df = df.sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features]
    y = df['target']
    
    # Split: Last 20% is the "Future" (Hold-out Test)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. Base Models (Optimized for Probability, not Class)
    # XGBoost with Multi-Softprob
    xgb_base = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=6,
        objective='multi:softprob', eval_metric='mlogloss', random_state=42
    )
    
    # Random Forest (Entropy criterion is better for probs)
    rf_base = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, criterion='log_loss', random_state=42
    )
    
    # 2. CALIBRATION (The Secret Sauce)
    # We use Isotonic calibration to fix overconfidence
    logging.info("Calibrating Models...")
    xgb_cal = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
    rf_cal = CalibratedClassifierCV(rf_base, method='isotonic', cv=3)
    
    # 3. Soft Voting Ensemble
    # We Average the probabilities (Soft Vote)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_cal), ('rf', rf_cal)],
        voting='soft'
    )
    
    logging.info("Training Ensemble...")
    ensemble.fit(X_train_s, y_train)
    
    # 4. Evaluation
    probs = ensemble.predict_proba(X_test_s)
    loss = log_loss(y_test, probs)
    acc = accuracy_score(y_test, ensemble.predict(X_test_s))
    
    # Progol Metric
    exp_hits = progol_metric(y_test, probs)
    
    logging.info(f"📊 RESULTS (Hold-out Test):")
    logging.info(f"   Log Loss: {loss:.4f} (Lower is better)")
    logging.info(f"   Accuracy: {acc:.4f}")
    logging.info(f"   Expected Correct Picks (per 14): {exp_hits:.2f}")
    
    # Save
    joblib.dump({'model': ensemble, 'scaler': scaler, 'features': features}, MODEL_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump({'log_loss': loss, 'accuracy': acc, 'exp_hits': exp_hits}, f)

if __name__ == "__main__":
    train_scientific_model()
