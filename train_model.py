import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import optuna
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': 500,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    model = xgb.XGBClassifier(**params, random_state=42, tree_method='hist')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[val_idx])
        scores.append(log_loss(y.iloc[val_idx], probs))
    return np.mean(scores)

def train_scientific_model():
    logging.info("🔬 STARTING BAYESIAN OPTIMIZATION & CALIBRATION")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    X, y = df[features], df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # 1. OPTUNA Study for XGBoost
    logging.info("Optimizing XGBoost with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_xgb(trial, X_scaled, y), n_trials=20)
    best_params = study.best_params
    
    # 2. Train Calibrated Ensemble
    xgb_best = xgb.XGBClassifier(**best_params, n_estimators=1000, random_state=42)
    rf_best = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42)
    
    logging.info("Calibrating Ensemble Components...")
    xgb_cal = CalibratedClassifierCV(xgb_best, method='isotonic', cv=5)
    rf_cal = CalibratedClassifierCV(rf_best, method='isotonic', cv=5)
    
    ensemble = VotingClassifier(estimators=[('xgb', xgb_cal), ('rf', rf_cal)], voting='soft')
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    ensemble.fit(X_train, y_train)
    
    # 3. Final Metrics
    probs = ensemble.predict_proba(X_test)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Extract importance using a proxy
    proxy_xgb = xgb.XGBClassifier(**best_params).fit(X_train, y_train)
    feat_imp = {f: float(i) for f, i in zip(features, proxy_xgb.feature_importances_)}

    metrics = {
        "accuracy": acc,
        "log_loss": log_loss(y_test, probs),
        "best_params": best_params,
        "feature_importance": feat_imp,
        "features": features,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    joblib.dump({'model': ensemble, 'scaler': scaler, 'features': features}, MODEL_PATH)
    logging.info(f"✅ PRODUCTION MODEL READY. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_scientific_model()
