import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import joblib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_PATH = 'data/processed/final_train_data.csv'
PRIMARY_MODEL_PATH = 'models/calibrated_ensemble.pkl'
UNDERDOG_MODEL_PATH = 'models/underdog_specialist.pkl'
METRICS_PATH = 'models/metrics.json'

def train_heavy_model():
    logging.info("🔬 STARTING MULTI-PERSPECTIVE TRAINING")
    df = pd.read_csv(DATA_PATH).sort_values('date')
    
    split_idx = int(len(df) * 0.85)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    # 1. ENCODE & SCALE
    encoder = TargetEncoder(cols=['venue', 'referee'])
    train_df_enc = encoder.fit_transform(train_df[['venue', 'referee']], train_df['target'])
    test_df_enc = encoder.transform(test_df[['venue', 'referee']])
    
    def prepare_X(base_df, enc_df):
        X = base_df.drop(columns=['fixture_id', 'date', 'target', 'venue', 'referee'])
        X['venue_enc'] = enc_df['venue']
        X['ref_enc'] = enc_df['referee']
        return X

    X_train, y_train = prepare_X(train_df, train_df_enc), train_df['target']
    X_test, y_test = prepare_X(test_df, test_df_enc), test_df['target']
    
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    
    # 2. MODELS
    base_models = [
        ('xgb', CalibratedClassifierCV(xgb.XGBClassifier(n_estimators=200), method='isotonic')),
        ('rf', CalibratedClassifierCV(RandomForestClassifier(n_estimators=200), method='isotonic'))
    ]
    
    # Primary
    p_stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3, stack_method='predict_proba')
    p_stack.fit(X_train_s, y_train)
    
    # Underdog (Non-Home)
    nh_idx = y_train != 0
    u_stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3, stack_method='predict_proba')
    u_stack.fit(X_train_s[nh_idx], y_train[nh_idx])

    # 3. METRICS GENERATION
    y_pred = p_stack.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract Feature Importance from a proxy XGBoost for the report
    proxy = xgb.XGBClassifier(n_estimators=100).fit(X_train_s, y_train)
    feat_imp = {f: float(i) for f, i in zip(X_train.columns, proxy.feature_importances_)}

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "feature_importance": feat_imp,
        "features": X_train.columns.tolist()
    }
    
    # 4. SAVE
    os.makedirs('models', exist_ok=True)
    with open(METRICS_PATH, 'w') as f: json.dump(metrics, f, indent=4)
    joblib.dump({'model': p_stack, 'scaler': scaler, 'encoder': encoder, 'features': X_train.columns.tolist()}, PRIMARY_MODEL_PATH)
    joblib.dump({'model': u_stack}, UNDERDOG_MODEL_PATH)
    
    print("\n" + "="*30)
    print(f"TRAINING COMPLETE")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Macro: {report['macro avg']['f1-score']:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    train_heavy_model()
