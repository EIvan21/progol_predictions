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
from scipy.optimize import minimize_scalar
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


def fit_temperature(y_prob, y_true, eps=1e-12):
    """Post-hoc Platt-style temperature scaling on the holdout set.

    Treats predict_proba output as already-softmaxed probs over 3 classes,
    converts to log-space pseudo-logits, then finds T in [0.1, 5.0] that
    minimizes NLL on the holdout. T<1 sharpens (more confident), T>1 softens.
    Returns (T_opt, log_loss_before, log_loss_after)."""
    y_prob = np.clip(y_prob, eps, 1.0)
    y_true = np.asarray(y_true)
    logits = np.log(y_prob)

    def nll(T):
        scaled = np.exp(logits / T)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        p_true = np.clip(scaled[np.arange(len(y_true)), y_true], eps, 1.0)
        return float(-np.mean(np.log(p_true)))

    res = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded',
                          options={'xatol': 1e-3})
    return float(res.x), nll(1.0), float(res.fun)


def apply_temperature(y_prob, T, eps=1e-12):
    if T is None or abs(T - 1.0) < 1e-3:
        return y_prob
    y_prob = np.clip(y_prob, eps, 1.0)
    scaled = np.exp(np.log(y_prob) / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


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

    # Class balance is already handled at the base-estimator level
    # (class_weight='balanced' on lgb/cat/rf, class_weights= on cat). Adding
    # sample_weight at stacking level + class_weight on the meta-LR triple-
    # weights the minority class, which produced a strong over-prediction
    # of draws (14X/7V/0L on a 21-match slate vs ~9/6/6 historical baseline).
    # Pipeline-wrapping additionally drops sample_weight before it reaches
    # the inner clf step (sklearn issue #21134), so the only places sw was
    # actually landing were the isotonic calibrator and the meta-LR — both
    # of which were biasing toward the central class. Drop both.
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method='predict_proba', n_jobs=-1,
    )

    logger.info("Training stacking ensemble...")
    stacking_model.fit(X_train, y_train)

    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    brier = calculate_brier_score(y_test.values, y_prob)
    loss = log_loss(y_test, y_prob, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred, output_dict=True)

    # Temperature scaling fit on the holdout. We tune T to minimize NLL,
    # then re-evaluate to confirm it actually helped (sanity check —
    # in rare cases the optimizer hits a boundary and post-loss > pre-loss).
    T_opt, loss_before_T, loss_after_T = fit_temperature(y_prob, y_test.values)
    if loss_after_T < loss_before_T:
        logger.info(f"Temperature: T={T_opt:.3f}, log_loss {loss_before_T:.4f} -> {loss_after_T:.4f}")
        y_prob_T = apply_temperature(y_prob, T_opt)
        loss_T = log_loss(y_test, y_prob_T, labels=[0, 1, 2])
        brier_T = calculate_brier_score(y_test.values, y_prob_T)
    else:
        logger.info(f"Temperature: T={T_opt:.3f} did not improve, disabling (T=1.0)")
        T_opt = 1.0
        loss_T, brier_T = loss, brier

    proxy = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    proxy_pipe = _build_base_pipeline(proxy)
    proxy_pipe.fit(X_train, y_train)
    feat_imp = {f: float(i) for f, i in zip(feature_cols, proxy_pipe.named_steps['clf'].feature_importances_)}

    metrics = {
        "accuracy": acc, "log_loss": loss, "brier_score": brier,
        "log_loss_temperature_scaled": loss_T,
        "brier_score_temperature_scaled": brier_T,
        "temperature": T_opt,
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

    payload = {
        'model': stacking_model,
        'features': feature_cols,
        'version': version_dir.name,
        'temperature': T_opt,
    }
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
    print(f"Version:        {version_dir.name}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Log Loss:       {loss:.4f}  ->  {loss_T:.4f} (T={T_opt:.3f})")
    print(f"Brier Score:    {brier:.4f}  ->  {brier_T:.4f}")
    print(f"F1-Macro:       {report['macro avg']['f1-score']:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    train_heavy_model()
