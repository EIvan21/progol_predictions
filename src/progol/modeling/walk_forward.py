"""Walk-forward validation: refits the model at N expanding-window origins
and reports per-fold accuracy / log-loss / Brier. Run as:

    python -m src.progol.modeling.walk_forward --folds 6
"""
import argparse
import json
import logging
from typing import List

import numpy as np
import pandas as pd

from src.progol import config
from src.progol.modeling.train import _build_base_pipeline, calculate_brier_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def _build_stacker():
    estimators = [
        ('lgb', CalibratedClassifierCV(_build_base_pipeline(
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
        ), method='isotonic', cv=3)),
        ('xgb', CalibratedClassifierCV(_build_base_pipeline(
            xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42, eval_metric='mlogloss')
        ), method='isotonic', cv=3)),
        ('cat', CalibratedClassifierCV(_build_base_pipeline(
            CatBoostClassifier(n_estimators=300, learning_rate=0.03, depth=6, random_state=42, verbose=0, allow_writing_files=False)
        ), method='isotonic', cv=3)),
        ('rf', CalibratedClassifierCV(_build_base_pipeline(
            RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        ), method='isotonic', cv=3)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method='predict_proba', n_jobs=-1,
    )


def walk_forward(df: pd.DataFrame, n_folds: int = 6, min_train_frac: float = 0.4) -> List[dict]:
    df = df.sort_values('date').dropna().reset_index(drop=True)
    feature_cols = config.CAT_COLS + config.FEATURE_COLS
    n = len(df)
    start = int(n * min_train_frac)
    fold_size = max(1, (n - start) // n_folds)

    results = []
    for i in range(n_folds):
        train_end = start + i * fold_size
        test_end = min(n, train_end + fold_size)
        if test_end <= train_end:
            break
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(test) < 10 or train['target'].nunique() < 3:
            continue

        Xtr, ytr = train[feature_cols], train['target']
        Xte, yte = test[feature_cols], test['target']
        # Time-decay weighting (matches train.py). Class-balanced weights
        # were previously used here but are no longer applied per the
        # over-prediction-of-draws investigation; class balance is handled
        # at the base estimator level (class_weight on lgb/cat/rf,
        # class_weights on cat). See train.py comments for details.
        train_dates = pd.to_datetime(train['date'])
        ref_date = train_dates.max()
        sw = np.exp(-((ref_date - train_dates).dt.days) / 365.0).values

        model = _build_stacker()
        model.fit(Xtr, ytr, sample_weight=sw)
        prob = model.predict_proba(Xte)
        pred = model.predict(Xte)

        fold = {
            'fold': i + 1,
            'train_size': len(train),
            'test_size': len(test),
            'train_end_date': str(train['date'].max()),
            'test_end_date': str(test['date'].max()),
            'accuracy': float(accuracy_score(yte, pred)),
            'log_loss': float(log_loss(yte, prob, labels=[0, 1, 2])),
            'brier_score': calculate_brier_score(yte.values, prob),
        }
        logger.info(f"Fold {fold['fold']} ✓  acc={fold['accuracy']:.3f} brier={fold['brier_score']:.3f}")
        results.append(fold)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--min-train-frac', type=float, default=0.4)
    parser.add_argument('--out', type=str, default=str(config.MODEL_DIR / 'walk_forward.json'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    df = pd.read_csv(config.TRAIN_CSV)
    folds = walk_forward(df, n_folds=args.folds, min_train_frac=args.min_train_frac)

    summary = {
        'folds': folds,
        'mean_accuracy': float(np.mean([f['accuracy'] for f in folds])) if folds else None,
        'mean_brier': float(np.mean([f['brier_score'] for f in folds])) if folds else None,
        'mean_log_loss': float(np.mean([f['log_loss'] for f in folds])) if folds else None,
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
