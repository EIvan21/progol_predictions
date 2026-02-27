import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
METRICS_PATH = 'models/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def get_model(model_type):
    if model_type == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42, early_stopping_rounds=50)
    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, class_weight='balanced')
    elif model_type == 'CatBoost':
        return CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, verbose=100, early_stopping_rounds=50)
    elif model_type == 'NeuralNetwork':
        # 3 Layers: 64 -> 32 -> 16 nodes
        return MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42,
            verbose=False
        )

def train_progol_model(df):
    model_type = os.getenv('MODEL_TYPE', 'XGBoost')
    logging.info(f"--- 🧠 TRAINING ARCHITECTURE: {model_type} ---")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name']
    features = [c for c in df.columns if c not in exclude]
    
    df = df.dropna(subset=features + ['target'])
    X = df[features]
    y = df['target']
    
    # NEURAL NETWORK REQUIREMENT: Feature Scaling
    if model_type == 'NeuralNetwork':
        logging.info("Applying Standard Scaling for Neural Network...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    model = get_model(model_type)
    
    if model_type == 'XGBoost':
        sw = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_test, y_test)], verbose=100)
    elif model_type == 'CatBoost':
        model.set_params(auto_class_weights='Balanced')
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    else:
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"VALIDATION ACCURACY: {test_acc:.4f} | F1: {f1:.4f}")

    metrics = {
        "model_type": model_type,
        "accuracy": test_acc,
        "f1_macro": f1,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": dict(zip(features, [float(x) for x in getattr(model, 'feature_importances_', np.zeros(len(features)))]))
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    logging.info(f"Model saved to {MODEL_PATH}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
    else: logging.error("Data missing.")
