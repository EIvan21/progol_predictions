import json
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, log_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.progol import config
from src.progol.utils import drift
from src.progol.utils.logging_setup import configure as configure_logging
from src.progol.storage import gcs, versioning

logger = logging.getLogger(__name__)


def calculate_brier_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    scores = []
    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        scores.append(brier_score_loss(y_bin, y_prob[:, i]))
    return float(np.mean(scores))


def _build_base_pipeline(estimator):
    """TargetEncoder + StandardScaler + estimator. Refits per-fold inside CV
    so the encoder doesn't see fold validation targets."""
    return Pipeline([
        ('encoder', TargetEncoder(cols=config.CAT_COLS)),
        ('scaler', StandardScaler()),
        ('clf', estimator),
    ])


def _load_best_lgb_params() -> dict:
    if not config.BEST_PARAMS_PATH.exists():
        return {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 31,
                'random_state': 42, 'verbose': -1}
    payload = json.loads(config.BEST_PARAMS_PATH.read_text())
    params = payload.get('best_params', {})
    params.setdefault('random_state', 42)
    params.setdefault('verbose', -1)
    return params


def train_heavy_model():
    configure_logging()
    logger.info("Time-based holdout training")
    df = pd.read_csv(config.TRAIN_CSV).sort_values('date').dropna()

    split_idx = int(len(df) * 0.85)
    train_full = df.iloc[:split_idx]
    test_holdout = df.iloc[split_idx:]
    logger.info(f"Train: {len(train_full)}  Holdout: {len(test_holdout)}")

    feature_cols = config.CAT_COLS + config.FEATURE_COLS
    X_train = train_full[feature_cols]
    y_train = train_full['target']
    X_test = test_holdout[feature_cols]
    y_test = test_holdout['target']

    lgb_params = _load_best_lgb_params()
    lgb_params.setdefault('class_weight', 'balanced')
    lgb_clf = lgb.LGBMClassifier(**lgb_params)

    # XGBoost has no class_weight param for multiclass; sample_weight is the lever.
    xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6,
                                random_state=42, eval_metric='mlogloss')

    # CatBoost: per-class weights derived from observed class frequencies.
    class_freq = y_train.value_counts(normalize=True).sort_index()
    cat_weights = (1.0 / class_freq).reindex([0, 1, 2]).fillna(1.0).tolist()
    cat_clf = CatBoostClassifier(n_estimators=300, learning_rate=0.03, depth=6,
                                 random_state=42, verbose=0, allow_writing_files=False,
                                 class_weights=cat_weights)

    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42,
                                    class_weight='balanced')

    # Calibration: isotonic. Sigmoid (Platt) was preferred for robustness on
    # small/medium data, but sklearn 1.5.x routes sigmoid through the Cython
    # CyHalfBinomialLoss whose dtype dispatch breaks across mixed-dtype proba
    # outputs (xgb→float32, lgb/cat/rf→float64). Isotonic (PAVA) is
    # dtype-agnostic. Watch for probability collapse on tail classes.
    estimators = [
        ('lgb', CalibratedClassifierCV(_build_base_pipeline(lgb_clf), method='isotonic', cv=3)),
        ('xgb', CalibratedClassifierCV(_build_base_pipeline(xgb_clf), method='isotonic', cv=3)),
        ('cat', CalibratedClassifierCV(_build_base_pipeline(cat_clf), method='isotonic', cv=3)),
        ('rf',  CalibratedClassifierCV(_build_base_pipeline(rf_clf),  method='isotonic', cv=3)),
    ]

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method='predict_proba', n_jobs=-1,
    )

    logger.info("Training stacking ensemble...")
    stacking_model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    brier = calculate_brier_score(y_test.values, y_prob)
    loss = log_loss(y_test, y_prob, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred, output_dict=True)

    proxy = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    proxy_pipe = _build_base_pipeline(proxy)
    proxy_pipe.fit(X_train, y_train)
    feat_imp = {f: float(i) for f, i in zip(feature_cols, proxy_pipe.named_steps['clf'].feature_importances_)}

    metrics = {
        "accuracy": acc, "log_loss": loss, "brier_score": brier,
        "classification_report": report, "feature_importance": feat_imp,
        "features": feature_cols,
        "lgb_params": lgb_params,
        "train_size": len(train_full), "holdout_size": len(test_holdout),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version_dir = versioning.new_version_dir(config.MODEL_DIR)

    with open(version_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(config.METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    payload = {'model': stacking_model, 'features': feature_cols, 'version': version_dir.name}
    joblib.dump(payload, version_dir / 'calibrated_ensemble.pkl')
    joblib.dump(payload, config.PRIMARY_MODEL_PATH)

    feature_stats = drift.fit_stats(train_full, config.FEATURE_COLS)
    drift.save_stats(feature_stats, config.FEATURE_STATS_PATH)
    drift.save_stats(feature_stats, version_dir / 'feature_stats.json')

    versioning.write_latest_pointer(config.MODEL_DIR, version_dir, metadata={
        'accuracy': acc, 'brier_score': brier, 'log_loss': loss,
    })

    # Python's google-auth fails on GCE Metadata Server mTLS in this image.
    # Local files are saved above; startup.sh's gsutil rsync (gcloud auth)
    # uploads them. Don't let an upload error abort the rest of the pipeline.
    try:
        gcs.upload_dir(config.MODEL_DIR, gcs_prefix='models', include_suffixes=['.pkl', '.json'])
    except Exception as e:
        logger.warning("Python GCS upload skipped (%s); bash gsutil rsync will sync.", e)

    print("\n" + "=" * 40)
    print(f"Version:     {version_dir.name}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"F1-Macro:    {report['macro avg']['f1-score']:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    train_heavy_model()
