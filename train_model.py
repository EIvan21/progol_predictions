import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
PRIMARY_MODEL_PATH = 'models/calibrated_ensemble.pkl'
METRICS_PATH = 'models/metrics.json'

def calculate_brier_score(y_true, y_prob):
    # Multi-class Brier score (average of per-class Brier scores)
    n_classes = y_prob.shape[1]
    brier_scores = []
    for i in range(n_classes):
        # Create binary target for class i
        y_binary = (y_true == i).astype(int)
        brier_scores.append(brier_score_loss(y_binary, y_prob[:, i]))
    return np.mean(brier_scores)

def train_heavy_model():
    logging.info("🔬 STARTING WALK-FORWARD VALIDATION TRAINING (WFV)")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    # Drop rows with NaN if any slipped through
    df = df.dropna()
    
    # 1. TIME-BASED SPLIT (Last 15% for final Holdout Test)
    split_idx = int(len(df) * 0.85)
    train_full = df.iloc[:split_idx]
    test_holdout = df.iloc[split_idx:]
    
    logging.info(f"Train Size: {len(train_full)} | Holdout Test Size: {len(test_holdout)}")

    # 2. FEATURE PREP
    # Define categorical columns that need Target Encoding
    cat_cols = ['venue', 'referee', 'league_id']
    drop_cols = ['fixture_id', 'date', 'target'] + cat_cols
    
    # Initialize Transformers
    encoder = TargetEncoder(cols=cat_cols)
    scaler = StandardScaler()
    
    # Fit Transformers on Full Training Set (Standard Procedure)
    # Note: In strict WFV, we would refit inside the loop, but for the final model we fit on all train data.
    X_train_raw = train_full.drop(columns=['fixture_id', 'date', 'target'])
    y_train = train_full['target']
    
    # Encode
    X_train_enc = encoder.fit_transform(X_train_raw, y_train)
    
    # Scale
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=X_train_enc.columns)
    
    # Prepare Holdout
    X_test_raw = test_holdout.drop(columns=['fixture_id', 'date', 'target'])
    y_test = test_holdout['target']
    X_test_enc = encoder.transform(X_test_raw)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_enc), columns=X_test_enc.columns)

    # 3. DEFINE MODEL ZOO
    # LightGBM (Fast & Accurate)
    lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    
    # XGBoost (The Classic)
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42, eval_metric='mlogloss')
    
    # CatBoost (Handles Categorical nuances well)
    cat_clf = CatBoostClassifier(n_estimators=300, learning_rate=0.03, depth=6, random_state=42, verbose=0, allow_writing_files=False)
    
    # Random Forest (Diversity)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

    # Base Learners with Calibration
    # We use Isotonic calibration to ensure probabilities are realistic
    estimators = [
        ('lgb', CalibratedClassifierCV(lgb_clf, method='isotonic', cv=3)),
        ('xgb', CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)),
        ('cat', CalibratedClassifierCV(cat_clf, method='isotonic', cv=3)),
        ('rf', CalibratedClassifierCV(rf_clf, method='isotonic', cv=3))
    ]

    # Meta-Learner: Logistic Regression works well to combine probabilities
    # We weight classes to handle the "Draw" imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=TimeSeriesSplit(n_splits=5), # CRITICAL: Walk-Forward Validation in Stacking
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    logging.info("🧠 Training Stacking Ensemble with TimeSeries Cross-Validation...")
    stacking_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # 4. EVALUATION
    logging.info("📊 Evaluating on Holdout Set...")
    y_pred = stacking_model.predict(X_test_scaled)
    y_prob = stacking_model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    brier = calculate_brier_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature Importance (Proxy from LGBM)
    proxy_model = lgb.LGBMClassifier(n_estimators=100).fit(X_train_scaled, y_train)
    feat_imp = {f: float(i) for f, i in zip(X_train_scaled.columns, proxy_model.feature_importances_)}
    
    metrics = {
        "accuracy": acc,
        "log_loss": loss,
        "brier_score": brier,
        "classification_report": report,
        "feature_importance": feat_imp,
        "features": X_train_scaled.columns.tolist()
    }

    # 5. SAVE ARTIFACTS
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    
    # Save the full pipeline components
    joblib.dump({
        'model': stacking_model,
        'scaler': scaler,
        'encoder': encoder,
        'features': X_train_scaled.columns.tolist()
    }, PRIMARY_MODEL_PATH)
    
    print("\n" + "="*40)
    print(f"🚀 TRAINING SUCCESSFUL (Walk-Forward Validated)")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Brier Score: {brier:.4f} (Lower is Better)")
    print(f"F1-Macro:    {report['macro avg']['f1-score']:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    train_heavy_model()
