import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

MODEL_PATH = 'models/progol_model.bin'
ENSEMBLE_PATH = 'models/ensemble_models.pkl'
METRICS_PATH = 'models/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def get_model(model_type):
    if model_type == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
    elif model_type == 'CatBoost':
        return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, random_seed=42, verbose=0)
    elif model_type == 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)

def train_progol_model(df):
    model_type = os.getenv('MODEL_TYPE', 'XGBoost')
    logging.info(f"--- 🧠 TRAINING ARCHITECTURE: {model_type} ---")
    
    # CRITICAL: Prevent Data Leakage by excluding goals and IDs
    exclude = [
        'fixture_id', 'date', 'target', 'home_id', 'away_id', 
        'home_name', 'away_name', 'status', 'league_name',
        'goals_home', 'goals_away', 'total_goals', 'result'
    ]
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=features + ['target'])
    X, y = df[features], df['target']
    
    logging.info(f"Training with {len(features)} Features: {features}")
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    
    models = {}
    if model_type == 'Ensemble':
        for m_name in ['XGBoost', 'RandomForest', 'CatBoost', 'NeuralNetwork']:
            logging.info(f"Training {m_name}...")
            m = get_model(m_name)
            m.fit(X_train, y_train)
            models[m_name] = m
        with open(ENSEMBLE_PATH, 'wb') as f: pickle.dump(models, f)
        probs = np.mean([m.predict_proba(X_test) for m in models.values()], axis=0)
        y_pred = np.argmax(probs, axis=1)
    else:
        model = get_model(model_type)
        model.fit(X_train, y_train)
        models[model_type] = model
        with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model_type": model_type,
        "accuracy": acc,
        "classification_report": report,
        "features": features,
        "best_params": "Standard Set"
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    logging.info(f"SUCCESS: Accuracy {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
