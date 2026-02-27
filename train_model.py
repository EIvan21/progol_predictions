import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import os
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'models/progol_stack_model.bin'
METRICS_PATH = 'models/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/processed/final_train_data.csv'

def train_progol_model(df):
    logging.info("--- 🏆 TRAINING STRATEGY 6: DIFFERENTIAL ENSEMBLE ---")
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=['target']); X = df[features].fillna(0); y = df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Complex Neural Network (Deep Learning)
    nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True, random_state=42)
    
    # 2. Calibrated Base Models (Fixes the "All L" confidence bias)
    xgb_model = CalibratedClassifierCV(xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=500), cv=cv)
    rf_model = CalibratedClassifierCV(RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced'), cv=cv)
    
    base_models = [('xgb', xgb_model), ('rf', rf_model), ('cat', CatBoostClassifier(iterations=500, silent=True, auto_class_weights='Balanced')), ('nn', nn)]
    
    # 3. Meta-Stacking
    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=cv, stack_method='predict_proba', n_jobs=-1)
    
    logging.info("Training Meta-Stacker with Calibration...")
    stack_model.fit(X_train, y_train)
    
    y_pred = stack_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract importance from XGB part of Stacker
    # Note: Accessing importance in a Stacker with Calibration is complex, saving placeholder
    metrics = {"model_type": "Strategy6_Differential", "accuracy": acc, "features": features, "classification_report": report, "feature_importance": {}}
    
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    with open(MODEL_PATH, 'wb') as f: pickle.dump(stack_model, f)
    
    print(f"\n--- 📈 PERFORMANCE: {acc:.4f} ---\n{classification_report(y_test, y_pred)}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH); train_progol_model(df)
