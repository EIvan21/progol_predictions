import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os
import json
import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)

METRICS_PATH = 'models/metrics.json'
ENSEMBLE_PATH = 'models/ensemble_models.pkl'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def tune_and_fit_elite(model_name, X, y, cv):
    logging.info(f"💎 ELITE TUNING: {model_name}...")
    if model_name == 'XGBoost':
        param = {'max_depth': [5, 7], 'learning_rate': [0.05, 0.1], 'n_estimators': [500]}
        grid = GridSearchCV(xgb.XGBClassifier(random_state=42), param, cv=cv, scoring='f1_macro', n_jobs=-1)
    elif model_name == 'RandomForest':
        param = {'max_depth': [12, 15], 'min_samples_leaf': [5, 10], 'n_estimators': [300]}
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param, cv=cv, scoring='f1_macro', n_jobs=-1)
    elif model_name == 'CatBoost':
        return CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, auto_class_weights='Balanced').fit(X, y)
    
    grid.fit(X, y)
    return grid.best_estimator_

def train_progol_model(df):
    model_type = os.getenv('MODEL_TYPE', 'Ensemble')
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away', 'year', 'total_goals', 'opp_id', 'is_win']
    features = [c for c in df.columns if c in ['league_id', 'venue', 'referee', 'roll_adj_gf_home', 'roll_adj_ga_home', 'clean_sheet_rate_home', 'power_score_home', 'days_rest_home', 'roll_adj_gf_away', 'roll_adj_ga_away', 'clean_sheet_rate_away', 'power_score_away', 'days_rest_away']]
    
    df = df.dropna(subset=features + ['target'])
    X, y = df[features], df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    
    # BALANCING: Use SMOTE to help with the "Draw" class
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logging.info(f"Balanced Dataset: {len(X_train_bal)} samples.")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {}
    
    if model_type == 'Ensemble':
        all_probs = []
        for m_name in ['XGBoost', 'RandomForest', 'CatBoost']:
            m = tune_and_fit_elite(m_name, X_train_bal, y_train_bal, cv)
            models[m_name] = m
            all_probs.append(m.predict_proba(X_test))
        with open(ENSEMBLE_PATH, 'wb') as f: pickle.dump(models, f)
        y_pred = np.argmax(np.mean(all_probs, axis=0), axis=1)
    else:
        model = tune_and_fit_elite(model_type, X_train_bal, y_train_bal, cv)
        model.fit(X_train_bal, y_train_bal)
        models[model_type] = model
        with open('models/progol_model.bin', 'wb') as f: pickle.dump(model, f)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    m_main = models.get('XGBoost') or list(models.values())[0]
    feat_imp = dict(zip(features, [float(x) for x in getattr(m_main, 'feature_importances_', np.zeros(len(features)))]))

    metrics = {"model_type": model_type, "accuracy": acc, "classification_report": report, "features": features, "feature_importance": feat_imp}
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    logging.info(f"💎 SUCCESS: Elite Accuracy {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
