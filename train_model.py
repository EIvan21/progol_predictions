import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def progol_metric(y_true, y_probs):
    n = len(y_true)
    correct_prob_sum = 0
    for i in range(n):
        true_class = int(y_true.iloc[i])
        correct_prob_sum += y_probs[i, true_class]
    return (correct_prob_sum / n) * 14

def train_scientific_model():
    logging.info("🔬 STARTING SCIENTIFIC CALIBRATION PIPELINE")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    X, y = df[features], df['target']
    
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    xgb_base = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    rf_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42)
    
    xgb_cal = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
    rf_cal = CalibratedClassifierCV(rf_base, method='isotonic', cv=3)
    
    ensemble = VotingClassifier(estimators=[('xgb', xgb_cal), ('rf', rf_cal)], voting='soft')
    ensemble.fit(X_train_s, y_train)
    
    # Extract Feature Importance for reporting (from a standalone XGB for proxy)
    report_xgb = xgb.XGBClassifier(n_estimators=100).fit(X_train_s, y_train)
    feat_imp = {f: float(i) for f, i in zip(features, report_xgb.feature_importances_)}

    probs = ensemble.predict_proba(X_test_s)
    y_pred = ensemble.predict(X_test_s)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, probs),
        "exp_hits": progol_metric(y_test, probs),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": feat_imp,
        "features": features
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    joblib.dump({'model': ensemble, 'scaler': scaler, 'features': features}, MODEL_PATH)
    logging.info(f"✅ SUCCESS: Exp Hits {metrics['exp_hits']:.2f}")

if __name__ == "__main__":
    train_scientific_model()
