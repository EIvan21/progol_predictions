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


def fit_dirichlet(y_prob, y_true, eps=1e-9):
    """Dirichlet (matrix) calibration on the holdout set.

    Treats log(y_prob) as a 3-d feature vector and fits a multinomial
    logistic regression to predict y_true. With low regularization this
    is the full Dirichlet calibrator (Kull et al. 2019); it can fix
    per-class miscalibration that a single scalar temperature cannot,
    e.g. systematic under-prediction of the draw class.

    Returns the fitted LogisticRegression. Use `apply_dirichlet` to
    transform raw probs at inference."""
    y_prob = np.clip(y_prob, eps, 1.0)
    log_probs = np.log(y_prob)
    y_true = np.asarray(y_true)
    # All 3 classes must be present; if not, fall back to identity.
    if len(np.unique(y_true)) < 3:
        return None
    # sklearn 1.5+ defaults to multinomial for multi-class; passing
    # multi_class kwarg now warns. C=1e4 is near-unregularized (full
    # Dirichlet matrix scaling). lbfgs handles multinomial natively.
    lr = LogisticRegression(max_iter=1000, C=1e4, solver='lbfgs')
    lr.fit(log_probs, y_true)
    # Reorder columns of coef/proba to match [0, 1, 2] in case classes_ differs.
    if not np.array_equal(lr.classes_, np.array([0, 1, 2])):
        return None
    return lr


def apply_dirichlet(y_prob, calibrator, eps=1e-9):
    if calibrator is None:
        return y_prob
    y_prob = np.clip(y_prob, eps, 1.0)
    log_probs = np.log(y_prob)
    return calibrator.predict_proba(log_probs)


def _build_base_pipeline(estimator):
    """TargetEncoder + StandardScaler + estimator. Refits per-fold inside CV
    so the encoder doesn't see fold validation targets.

    smoothing=10 shrinks rare-category target means toward the global mean
    more aggressively than the library default (1.0). With CAT_COLS now
    carrying home_id (~1342 teams) and referee (thousands, most with <10
    matches), the higher smoothing prevents single-match outliers from
    leaking as strong-signal encodings."""
    return Pipeline([
        ('encoder', TargetEncoder(cols=config.CAT_COLS, smoothing=10.0)),
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

    # Time-decay sample weight: exp(-Δdays / half_life). With half_life=365
    # a match from 1 year ago weighs ~37%, 2 years ago ~14%. This biases
    # the model toward recent form, transfers, and tactical trends in a
    # data set that spans 2019-2026.
    #
    # CAVEAT (sklearn #21134): Pipeline drops sample_weight before it hits
    # the inner clf step, so this currently only lands on the isotonic
    # calibrator and the meta-LR. Unlike the previous class-weight
    # experiment, time-decay weights are NOT correlated with the target
    # class, so they shouldn't induce the over-prediction-of-draws bias
    # that prior sample_weight attempts caused. Batch C plans a restructure
    # that pushes the weight to the base learners directly.
    train_dates = pd.to_datetime(train_full['date'])
    ref_date = train_dates.max()
    HALF_LIFE_DAYS = 365.0
    sample_weight = np.exp(-((ref_date - train_dates).dt.days) / HALF_LIFE_DAYS).values

    logger.info("Training stacking ensemble...")
    stacking_model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    brier = calculate_brier_score(y_test.values, y_prob)
    loss = log_loss(y_test, y_prob, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred, output_dict=True)

    # Dirichlet (matrix) calibration on the holdout — fixes per-class
    # miscalibration that scalar temperature cannot. Specifically targets
    # the draw class, which the stacked ensemble systematically under-
    # predicts even after isotonic calibration on each base estimator.
    # If Dirichlet fails to improve log-loss vs raw, fall back to scalar
    # temperature (older behavior); if both fail, identity.
    dirichlet_cal = fit_dirichlet(y_prob, y_test.values)
    loss_dir = float('inf')
    brier_dir = float('inf')
    y_prob_dir = None
    if dirichlet_cal is not None:
        y_prob_dir = apply_dirichlet(y_prob, dirichlet_cal)
        loss_dir = log_loss(y_test, y_prob_dir, labels=[0, 1, 2])
        brier_dir = calculate_brier_score(y_test.values, y_prob_dir)

    T_opt, loss_before_T, loss_after_T = fit_temperature(y_prob, y_test.values)

    if loss_dir < loss and loss_dir <= loss_after_T:
        logger.info(f"Calibration: Dirichlet, log_loss {loss:.4f} -> {loss_dir:.4f}")
        loss_T, brier_T = loss_dir, brier_dir
        T_opt = 1.0
        active_calibration = 'dirichlet'
    elif loss_after_T < loss_before_T:
        logger.info(f"Calibration: Temperature T={T_opt:.3f}, log_loss {loss:.4f} -> {loss_after_T:.4f}")
        y_prob_T = apply_temperature(y_prob, T_opt)
        loss_T = log_loss(y_test, y_prob_T, labels=[0, 1, 2])
        brier_T = calculate_brier_score(y_test.values, y_prob_T)
        dirichlet_cal = None
        active_calibration = 'temperature'
    else:
        logger.info(f"Calibration: neither Dirichlet nor T improved, disabled")
        T_opt = 1.0
        dirichlet_cal = None
        loss_T, brier_T = loss, brier
        active_calibration = 'none'

    proxy = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    proxy_pipe = _build_base_pipeline(proxy)
    proxy_pipe.fit(X_train, y_train)
    feat_imp = {f: float(i) for f, i in zip(feature_cols, proxy_pipe.named_steps['clf'].feature_importances_)}

    metrics = {
        "accuracy": acc, "log_loss": loss, "brier_score": brier,
        "log_loss_calibrated": loss_T,
        "brier_score_calibrated": brier_T,
        "temperature": T_opt,
        "calibration": active_calibration,
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
        'dirichlet_calibrator': dirichlet_cal,
        'calibration': active_calibration,
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
