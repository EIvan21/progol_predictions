import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import os
import json
import logging

# Setup Verbose Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)

MODEL_PATH = 'models/progol_xgb_model.json'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/processed/final_train_data.csv'

def train_progol_model(df):
    logging.info("--- 🧠 STARTING MODEL INTEGRITY CHECK ---")
    
    # 1. Verify Data Quantity
    total_rows = len(df)
    exclude = ['fixture_id', 'date', 'target', 'home_id', 'away_id', 'home_name', 'away_name', 'status', 'league_name']
    features = [c for c in df.columns if c not in exclude]
    
    logging.info(f"TOTAL DATASET SIZE: {total_rows} matches.")
    logging.info(f"FEATURES DETECTED ({len(features)}): {features}")
    
    # Drop rows with NaN (usually first games of a team's history)
    df = df.dropna(subset=features + ['target'])
    logging.info(f"CLEANED DATASET SIZE (Post-NaN removal): {len(df)} matches.")

    X = df[features]
    y = df['target']
    
    # 2. Train/Test Split with Stratification (Prevents Overfitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    logging.info(f"FINAL TRAINING POOL: {len(X_train)} matches.")
    logging.info(f"FINAL VALIDATION POOL: {len(X_test)} matches.")

    # 3. Handle Class Imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 4. XGBoost with EARLY STOPPING (The Anti-Overfit Shield)
    # We use a lower learning rate and higher estimators with stopping
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03, # Slower learning = better generalization
        max_depth=6,        # Limited depth to prevent memorization
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,          # Minimum loss reduction to split
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50 # Stop if no improvement for 50 iterations
    )
    
    logging.info("--- 🚀 TRAINING IN PROGRESS (With Real-Time Evaluation) ---")
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100 # Print status every 100 trees
    )
    
    # 5. Evaluate Results
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info("--- ✅ TRAINING COMPLETE ---")
    logging.info(f"TRAIN ACCURACY: {train_acc:.4f}")
    logging.info(f"VALIDATION ACCURACY: {test_acc:.4f}")
    
    # Detection of Overfitting
    gap = train_acc - test_acc
    if gap > 0.15:
        logging.warning(f"⚠️ ALERT: HIGH OVERFITTING DETECTED (Gap: {gap:.2f})")
    else:
        logging.info(f"✨ Model is generalizing well (Gap: {gap:.2f})")

    metrics = {
        "accuracy": test_acc,
        "f1_macro": f1,
        "train_acc": train_acc,
        "best_iteration": model.best_iteration,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": dict(zip(features, model.feature_importances_.tolist())),
        "best_params": model.get_params()
    }
    
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    model.save_model(MODEL_PATH)
    logging.info(f"Best Model saved. Validation Accuracy: {test_acc:.4f}")
    return metrics

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        train_progol_model(df)
    else:
        logging.error("Data file missing. Preprocess first.")
