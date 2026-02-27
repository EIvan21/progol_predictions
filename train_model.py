import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import os
import json
import logging
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)

MODEL_PATH = 'models/progol_xgb_model.json'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/processed/final_train_data.csv'

def train_progol_model(df):
    logging.info("Determining features from dataset...")
    
    # Dynamically select features based on what preprocess.py generated
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name']
    features = [c for c in df.columns if c not in exclude]
    
    df = df.dropna(subset=features + ['target'])
    X = df[features]
    y = df['target']
    
    logging.info(f"Training with {len(features)} features: {features}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Grid search parameters
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'gamma': [0.1, 0.2]
    }
    
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=1000,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    logging.info("Starting Hyperparameter Optimization...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    metrics = {
        "best_params": grid_search.best_params_,
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": dict(zip(features, best_model.feature_importances_.tolist()))
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    best_model.save_model(MODEL_PATH)
    logging.info(f"Model saved. Final Accuracy: {acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        train_progol_model(df)
    else:
        logging.error("Final training data not found.")
