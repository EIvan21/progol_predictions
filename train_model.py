import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import optuna
import os
import json
import logging
import pickle
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRICS_PATH = 'models/metrics.json'
MODEL_PATH = 'models/progol_stack_model.bin'
ENSEMBLE_PATH = 'models/ensemble_models.pkl'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def objective_xgb(trial, X, y):
    param = {'max_depth': trial.suggest_int('max_depth', 3, 7), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'n_estimators': 200}
    model = xgb.XGBClassifier(**param, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for t_idx, v_idx in cv.split(X, y):
        model.fit(X.iloc[t_idx], y.iloc[t_idx])
        scores.append(accuracy_score(y.iloc[v_idx], model.predict(X.iloc[v_idx])))
    return np.mean(scores)

def train_progol_model(df):
    model_type = config.MODEL_TYPE
    logging.info(f"--- 🏆 STARTING TRAINING: {model_type} ({'LOCAL' if config.IS_LOCAL_TEST else 'PROD'}) ---")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away', 'year', 'total_goals', 'result']
    features = [c for c in df.columns if c not in exclude]
    
    df = df.dropna(subset=['target'])
    X, y = df[features].fillna(0), df['target']
    
    logging.info(f"Dataset Size: {len(X)} rows | Features: {len(features)}")
    
    if len(X) < 10:
        logging.error("NOT ENOUGH DATA TO TRAIN.")
        return

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    # Force Balance for ALL production/test runs to ensure diverse L/E/V predictions
    rus = RandomUnderSampler(random_state=42)
    X_bal, y_bal = rus.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal)

    if model_type == 'Ensemble':
        logging.info("Optimizing XGBoost component...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective_xgb(t, X_train, y_train), n_trials=10)
        
        base = [
            ('xgb', xgb.XGBClassifier(**study.best_params, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('cat', CatBoostClassifier(iterations=300, silent=True))
        ]
        model = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=3, stack_method='predict_proba', n_jobs=-1)
        model.fit(X_train, y_train)
        # Importance from XGB
        temp_xgb = xgb.XGBClassifier(**study.best_params).fit(X_train, y_train)
        feat_imp = dict(zip(features, [float(x) for x in temp_xgb.feature_importances_]))
    else:
        # Fallback to simple XGB if not Ensemble
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05).fit(X_train, y_train)
        feat_imp = dict(zip(features, [float(x) for x in model.feature_importances_]))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {"model_type": model_type, "accuracy": acc, "feature_importance": feat_imp, "classification_report": report, "features": features}
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    
    logging.info(f"✅ SUCCESS: Accuracy {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
