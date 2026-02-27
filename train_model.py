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
import os
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

METRICS_PATH = 'models/metrics.json'
MODEL_PATH = 'models/progol_stack_model.bin'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def train_progol_model(df):
    logging.info("--- ⚖️ STARTING BIAS-CONTROLLED TRAINING ---")
    
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name', 'goals_home', 'goals_away', 'total_goals', 'result', 'year', 'venue', 'referee']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=['target'])
    X, y = df[features].fillna(0), df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    # 1. ENHANCED UNDER-SAMPLING: We ensure Draw and Away are perfectly represented
    rus = RandomUnderSampler(random_state=42)
    X_bal, y_bal = rus.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal)

    # 2. COST-SENSITIVE MODELS
    # We use 'balanced' weights to force the model to care about Draws
    base_models = [
        ('xgb', xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=300, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
        ('cat', CatBoostClassifier(iterations=300, silent=True, auto_class_weights='Balanced'))
    ]
    
    # 3. MULTINOMIAL META-LEARNER
    # This model learns how to combine the 3 brains to avoid the 'L' trap
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(multi_class='multinomial', class_weight={0: 0.8, 1: 1.2, 2: 1.0}),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    logging.info("Training Bias-Killer Stacker...")
    stack_model.fit(X_train, y_train)
    
    y_pred = stack_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "model_type": "BiasControlled_Stacker",
        "accuracy": acc,
        "features": features,
        "classification_report": report
    }
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(stack_model, f)
    
    print(f"\n--- 📈 BALANCED PERFORMANCE: {acc:.4f} ---")
    print(classification_report(y_test, y_pred))
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
