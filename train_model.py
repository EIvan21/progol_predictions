import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import optuna
import joblib
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("logs/heavy_training.log"), logging.StreamHandler()]
)

DATA_PATH = 'data/processed/final_train_data.csv'
MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def objective_ensemble(trial, X, y):
    xgb_params = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
    }
    model = xgb.XGBClassifier(**xgb_params, random_state=42, tree_method='hist')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(log_loss(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])))
    return np.mean(scores)

def train_heavy_model():
    logging.info("🔥 STARTING ROBUST INDUSTRIAL OPTIMIZATION")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    exclude = ['fixture_id', 'date', 'target']
    features = [c for c in df.columns if c not in exclude]
    
    # 1. Clean Data: Fill ALL NaNs before any processing
    X = df[features].fillna(0)
    y = df['target']
    
    # 2. Pipeline: Impute -> Scale
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    scaler = StandardScaler()
    
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=features)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=features)
    
    # 3. Bayesian Search
    logging.info("Optimizing Hyperparameters...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective_ensemble(t, X_scaled, y), n_trials=30) # Reduced trials for second attempt
    best = study.best_params

    # 4. Define Component Brains
    nn = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True, random_state=42)
    xgb_f = xgb.XGBClassifier(**best, n_estimators=500, random_state=42)
    rf_f = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

    # 5. Calibrate & Stack
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = [
        ('xgb', CalibratedClassifierCV(xgb_f, method='isotonic', cv=cv)),
        ('rf', CalibratedClassifierCV(rf_f, method='isotonic', cv=cv)),
        ('nn', CalibratedClassifierCV(nn, method='isotonic', cv=cv))
    ]
    
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=cv,
        stack_method='predict_proba',
        n_jobs=-1
    )

    logging.info("Training Final Ensemble (Safe from NaNs)...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
    stack_model.fit(X_train, y_train)
    
    probs = stack_model.predict_proba(X_test)
    acc = accuracy_score(y_test, stack_model.predict(X_test))
    
    logging.info(f"✅ SUCCESS: Accuracy {acc:.4f} | LogLoss {log_loss(y_test, probs):.4f}")

    metrics = {
        "accuracy": acc,
        "log_loss": log_loss(y_test, probs),
        "features": features,
        "classification_report": classification_report(y_test, stack_model.predict(X_test), output_dict=True)
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    joblib.dump({'model': stack_model, 'scaler': scaler, 'imputer': imputer, 'features': features}, MODEL_PATH)

if __name__ == "__main__":
    train_heavy_model()
