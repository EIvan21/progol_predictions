import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
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

def tune_and_fit_deep(model_name, X, y, cv):
    logging.info(f"🚀 DEEP TUNING for {model_name}...")
    
    if model_name == 'XGBoost':
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.07],
            'n_estimators': [1000],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        }
        grid = GridSearchCV(xgb.XGBClassifier(random_state=42, tree_method='hist'), param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    elif model_name == 'RandomForest':
        param_grid = {
            'max_depth': [10, 15, 20],
            'min_samples_leaf': [2, 5, 10],
            'n_estimators': [500],
            'max_features': ['sqrt', 'log2']
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    elif model_name == 'CatBoost':
        # CatBoost has a very powerful internal tuner
        return CatBoostClassifier(iterations=2000, learning_rate=0.02, depth=7, verbose=200, auto_class_weights='Balanced', early_stopping_rounds=50).fit(X, y)

    elif model_name == 'NeuralNetwork':
        # Deep MLP with more layers
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, early_stopping=True, random_state=42).fit(X, y)

    grid.fit(X, y)
    logging.info(f"🏆 Best Params: {grid.best_params_}")
    return grid.best_estimator_

def train_progol_model(df):
    model_type = os.getenv('MODEL_TYPE', 'Ensemble')
    logging.info(f"--- 🔥 HEAVY PRODUCTION TRAINING: {model_type} ---")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=features + ['target'])
    X, y = df[features], df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    
    # 5-FOLD Stratified CV for maximum reliability
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {}
    if model_type == 'Ensemble':
        for m_name in ['XGBoost', 'RandomForest', 'CatBoost']:
            models[m_name] = tune_and_fit_deep(m_name, X_train, y_train, cv)
        
        with open(ENSEMBLE_PATH, 'wb') as f: pickle.dump(models, f)
        probs = np.mean([m.predict_proba(X_test) for m in models.values()], axis=0)
        y_pred = np.argmax(probs, axis=1)
    else:
        model = tune_and_fit_deep(model_type, X_train, y_train, cv)
        models[model_type] = model
        with open('models/progol_model.bin', 'wb') as f: pickle.dump(model, f)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model_type": model_type,
        "accuracy": acc,
        "classification_report": report,
        "features": features
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    logging.info(f"🔥 FINAL PRODUCTION ACCURACY: {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
