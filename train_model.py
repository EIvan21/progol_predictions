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
    
    # Exclude IDs and target results to prevent leakage
    exclude = [
        'fixture_id', 'date', 'target', 'home_id', 'away_id', 
        'home_name', 'away_name', 'status', 'league_name',
        'goals_home', 'goals_away'
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
    feat_importance = {}

    if model_type == 'Ensemble':
        all_probs = []
        for m_name in ['XGBoost', 'RandomForest', 'CatBoost']:
            logging.info(f"Training {m_name}...")
            m = get_model(m_name)
            m.fit(X_train, y_train)
            models[m_name] = m
            all_probs.append(m.predict_proba(X_test))
            # Save importance from the first tree model (XGBoost)
            if m_name == 'XGBoost':
                feat_importance = dict(zip(features, [float(x) for x in m.feature_importances_]))
        
        with open(ENSEMBLE_PATH, 'wb') as f: pickle.dump(models, f)
        final_probs = np.mean(all_probs, axis=0)
        y_pred = np.argmax(final_probs, axis=1)
    else:
        model = get_model(model_type)
        model.fit(X_train, y_train)
        models[model_type] = model
        with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
        y_pred = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            feat_importance = dict(zip(features, [float(x) for x in model.feature_importances_]))

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model_type": model_type,
        "accuracy": acc,
        "classification_report": report,
        "features": features,
        "feature_importance": feat_importance
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    logging.info(f"SUCCESS: Accuracy {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
