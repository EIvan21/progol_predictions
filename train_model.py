import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna
import os
import json
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)

MODEL_PATH = 'models/progol_stack_model.bin'
METRICS_PATH = 'models/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def objective_xgb(trial, X, y):
    param = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
    }
    model = xgb.XGBClassifier(**param, random_state=42)
    # Use TimeSeriesSplit for realistic sports validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_t, y_t)
        scores.append(accuracy_score(y_v, model.predict(X_v)))
    return np.mean(scores)

def train_progol_model(df):
    logging.info("--- 🏆 STARTING HYPER-ENSEMBLE STACKING ---")
    
    features = ['league_id', 'league_ha_factor', 'venue_encoded', 'ref_encoded', 
                'roll_gf_home', 'roll_ga_home', 'cs_rate_home', 'power_score_home',
                'roll_gf_away', 'roll_ga_away', 'cs_rate_away', 'power_score_away']
    
    df = df.dropna(subset=features + ['target'])
    X, y = df[features], df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    # 1. OPTUNA BAYESIAN OPTIMIZATION (For the primary model)
    logging.info("Optimizing XGBoost via Bayesian Search...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_xgb(trial, X_scaled, y), n_trials=20)
    best_params = study.best_params
    logging.info(f"Best XGB Params: {best_params}")

    # 2. MODEL STACKING (The ultimate ensemble)
    base_models = [
        ('xgb', xgb.XGBClassifier(**best_params, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)),
        ('cat', CatBoostClassifier(iterations=500, silent=True, auto_class_weights='Balanced'))
    ]
    
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    logging.info("Training Meta-Stacker...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, shuffle=False)
    stack_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, stack_model.predict(X_test))
    logging.info(f"🔥 FINAL STACKED ACCURACY: {acc:.4f}")
    
    metrics = {
        "model_type": "StackedEnsemble",
        "accuracy": acc,
        "best_xgb_params": best_params,
        "features": features
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(stack_model, f)
    
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
