import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import optuna
import os
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'models/progol_stack_model.bin'
METRICS_PATH = 'models/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def objective_xgb(trial, X, y):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'n_estimators': 200,
        'tree_method': 'hist'
    }
    model = xgb.XGBClassifier(**param, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_t, y_t)
        scores.append(accuracy_score(y_v, model.predict(X_v)))
    return np.mean(scores)

def train_progol_model(df):
    logging.info("--- 🏆 STARTING ROBUST HYPER-ENSEMBLE ---")
    
    # Exclude text-based columns and IDs
    exclude = [
        'fixture_id', 'date', 'target', 'home_id', 'away_id', 
        'home_name', 'away_name', 'status', 'league_name',
        'goals_home', 'goals_away', 'total_goals', 'result', 'year',
        'venue', 'referee' # Added these to fix the crash
    ]
    features = [c for c in df.columns if c not in exclude]
    
    # 1. Smarter Cleaning: Only drop if Target is missing. 
    # Features missing values will be filled with 0.
    df = df.dropna(subset=['target'])
    X = df[features].fillna(0)
    y = df['target']
    
    logging.info(f"Initial Rows: {len(df)} | Features: {len(features)}")
    
    if len(X) == 0:
        logging.error("CRITICAL: No data remaining after cleaning!")
        return

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    # Optuna Study
    logging.info("Optimizing XGBoost...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_xgb(trial, X_scaled, y), n_trials=10)
    
    base_models = [
        ('xgb', xgb.XGBClassifier(**study.best_params, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')),
        ('cat', CatBoostClassifier(iterations=300, silent=True, auto_class_weights='Balanced'))
    ]
    
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    logging.info(f"Training on {len(X_train)} matches...")
    stack_model.fit(X_train, y_train)
    
    y_pred = stack_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"🔥 FINAL STACKED ACCURACY: {acc:.4f}")
    
    metrics = {
        "model_type": "RobustStackedEnsemble",
        "accuracy": acc,
        "features": features,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(stack_model, f)
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
