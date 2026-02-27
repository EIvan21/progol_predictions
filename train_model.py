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

METRICS_PATH = 'models/metrics.json'
MODEL_PATH = 'models/progol_stack_model.bin'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def objective_xgb(trial, X, y):
    param = {'max_depth': trial.suggest_int('max_depth', 3, 8), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'subsample': trial.suggest_float('subsample', 0.6, 0.9), 'n_estimators': 200, 'tree_method': 'hist'}
    model = xgb.XGBClassifier(**param, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]; y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_t, y_t); scores.append(accuracy_score(y_v, model.predict(X_v)))
    return np.mean(scores)

def train_progol_model(df):
    logging.info("--- 🏆 STARTING FINAL PRODUCTION TRAINING ---")
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away', 'total_goals', 'result', 'year', 'venue', 'referee']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=['target']); X = df[features].fillna(0); y = df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_xgb(trial, X_scaled, y), n_trials=10)
    
    # Base Models
    best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42)
    base_models = [
        ('xgb', best_xgb),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight='balanced')),
        ('cat', CatBoostClassifier(iterations=500, silent=True, auto_class_weights='Balanced'))
    ]
    
    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), stack_method='predict_proba', n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    stack_model.fit(X_train, y_train)
    
    # IMPORTANT: Re-train the tuned XGBoost on full set to get Importance for the report
    best_xgb.fit(X_train, y_train)
    feat_imp = {f: float(i) for f, i in zip(features, best_xgb.feature_importances_)}

    y_pred = stack_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model_type": "StackedEnsemble_Final",
        "accuracy": acc,
        "features": features,
        "feature_importance": feat_imp,
        "classification_report": report,
        "best_xgb_params": study.best_params
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(stack_model, f)
    
    print("\n" + "="*30)
    print(f"MODEL PERFORMANCE METRICS")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Macro:  {report['macro avg']['f1-score']:.4f}")
    print("="*30 + "\n")
    
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
