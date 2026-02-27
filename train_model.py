import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import optuna
import joblib
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("logs/heavy_training.log"), logging.StreamHandler()]
)

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def objective_ensemble(trial, X, y):
    # --- XGBoost Deep Tuning ---
    xgb_params = {
        'n_estimators': 500,
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('xgb_lr', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('xgb_min_child', 1, 10),
    }
    
    # --- Random Forest Deep Tuning ---
    rf_params = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('rf_max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('rf_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_leaf', 1, 10),
        'max_features': trial.suggest_categorical('rf_feat', ['sqrt', 'log2', None])
    }

    # Cross-Validation Strategy: 10-FOLD STRATIFIED
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # We evaluate based on a simplified ensemble for speed during tuning
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        
        m1 = xgb.XGBClassifier(**xgb_params, random_state=42, tree_method='hist')
        m1.fit(X_t, y_t)
        
        probs = m1.predict_proba(X_v)
        scores.append(log_loss(y_v, probs))
        
    return np.mean(scores)

def train_heavy_model():
    logging.info("🔥 INITIALIZING INDUSTRIAL-STRENGTH OPTIMIZATION PIPELINE")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    X, y = df[features], df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # 1. MASSIVE BAYESIAN SEARCH (100 Trials)
    logging.info("🚀 Starting 100-Trial Bayesian Search. This will take time...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_ensemble(trial, X_scaled, y), n_trials=100)
    
    best = study.best_params
    logging.info(f"🏆 SEARCH COMPLETE. Best LogLoss: {study.best_value:.4f}")

    # 2. CONSTRUCT DEEP MLP (The Robust Brain)
    nn = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    # 3. CONSTRUCT ENSEMBLE WITH CALIBRATION
    xgb_final = xgb.XGBClassifier(
        max_depth=best['xgb_max_depth'],
        learning_rate=best['xgb_lr'],
        subsample=best['xgb_subsample'],
        colsample_bytree=best['xgb_colsample'],
        reg_alpha=best['xgb_alpha'],
        reg_lambda=best['xgb_lambda'],
        min_child_weight=best['xgb_min_child'],
        n_estimators=2000,
        random_state=42
    )
    
    rf_final = RandomForestClassifier(
        max_depth=best['rf_max_depth'],
        min_samples_split=best['rf_split'],
        min_samples_leaf=best['rf_leaf'],
        max_features=best['rf_feat'],
        n_estimators=1000,
        random_state=42
    )

    # Calibrate every model individually before stacking
    logging.info("Calibrating Component Brains (10-Fold Isotonic)...")
    xgb_cal = CalibratedClassifierCV(xgb_final, method='isotonic', cv=10)
    rf_cal = CalibratedClassifierCV(rf_final, method='isotonic', cv=10)
    nn_cal = CalibratedClassifierCV(nn, method='isotonic', cv=10)

    # 4. FINAL META-STACKER
    base_models = [('xgb', xgb_cal), ('rf', rf_cal), ('nn', nn_cal)]
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(multi_class='multinomial'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )

    logging.info("Training Final Stacked Ensemble. Please Wait...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    stack_model.fit(X_train, y_train)
    
    # 5. VERIFICATION
    probs = stack_model.predict_proba(X_test)
    y_pred = stack_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logging.info(f"✅ HEAVY TRAINING COMPLETE.")
    logging.info(f"   Accuracy: {acc:.4f}")
    logging.info(f"   Log Loss: {log_loss(y_test, probs):.4f}")

    metrics = {
        "accuracy": acc,
        "log_loss": log_loss(y_test, probs),
        "best_params": best,
        "features": features,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    joblib.dump({'model': stack_model, 'scaler': scaler, 'features': features}, MODEL_PATH)

if __name__ == "__main__":
    train_heavy_model()
