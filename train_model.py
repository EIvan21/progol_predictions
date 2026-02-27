import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)

MODEL_PATH = 'models/progol_xgb_model.json'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/processed/final_train_data.csv'

def train_progol_model(df):
    logging.info("Starting Advanced Training with EWMA features...")
    
    # Updated feature set
    features = [
        'league_id', 'venue', 'referee',
        'ewm_goals_for_home', 'ewm_goals_against_home', 'ewm_goal_diff_home', 'ewm_form_points_home', 'days_rest_home',
        'ewm_goals_for_away', 'ewm_goals_against_away', 'ewm_goal_diff_away', 'ewm_form_points_away', 'days_rest_away'
    ]
    
    df = df.dropna(subset=features + ['target'])
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'gamma': [0.1, 0.2]
    }
    
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=1000,
        random_state=42,
        eval_metric='mlogloss',
        # Enable GPU if available (XGBoost handles this check)
        tree_method='hist' # Change to 'gpu_hist' if GPU is confirmed
    )
    
    logging.info("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
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
    logging.info(f"Model saved. Final Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        train_progol_model(df)
    else:
        logging.error("Processed data missing.")
