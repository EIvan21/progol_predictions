import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
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
        return xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    elif model_type == 'CatBoost':
        return CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
    elif model_type == 'NeuralNetwork':
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)

def train_progol_model(df):
    model_type = os.getenv('MODEL_TYPE', 'XGBoost')
    logging.info(f"--- 🧠 TRAINING ARCHITECTURE: {model_type} ---")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=features + ['target'])
    X, y = df[features], df['target']
    
    # Scaling for NN and consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    
    if model_type == 'Ensemble':
        logging.info("Training Ensemble (XGB, RF, Cat, NN)...")
        models = {}
        for m_name in ['XGBoost', 'RandomForest', 'CatBoost', 'NeuralNetwork']:
            logging.info(f"Training {m_name} part of ensemble...")
            m = get_model(m_name)
            m.fit(X_train, y_train)
            models[m_name] = m
        
        with open(ENSEMBLE_PATH, 'wb') as f:
            pickle.dump(models, f)
        logging.info(f"Ensemble saved to {ENSEMBLE_PATH}")
        # Metrics based on simple average
        probs = np.mean([m.predict_proba(X_test) for m in models.values()], axis=0)
        y_pred = np.argmax(probs, axis=1)
    else:
        model = get_model(model_type)
        model.fit(X_train, y_train)
        with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"FINAL ACCURACY: {acc:.4f}")
    
    metrics = {"model_type": model_type, "accuracy": acc}
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
    else: logging.error("Data missing.")
