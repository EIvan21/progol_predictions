"""Optuna hyperparameter tuning for the LightGBM base learner.

Uses a chronological 80/20 split inside the training set and minimizes log_loss.
Saves best params to models/best_params.json; train.py reads it if present.

    python -m src.progol.modeling.tune --trials 40 --timeout 1800
"""
import argparse
import json
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

import lightgbm as lgb
from src.progol import config
from src.progol.modeling.train import _build_base_pipeline

logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 96),
        'max_depth': trial.suggest_int('max_depth', -1, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42, 'verbose': -1,
    }
    model = _build_base_pipeline(lgb.LGBMClassifier(**params))
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)
    return log_loss(y_val, prob, labels=[0, 1, 2])


def tune(n_trials: int = 40, timeout: int = 1800) -> dict:
    df = pd.read_csv(config.TRAIN_CSV).sort_values('date').dropna()
    feature_cols = config.CAT_COLS + config.FEATURE_COLS
    split = int(len(df) * 0.85 * 0.85)
    X_train, y_train = df.iloc[:split][feature_cols], df.iloc[:split]['target']
    X_val, y_val = df.iloc[split:int(len(df) * 0.85)][feature_cols], df.iloc[split:int(len(df) * 0.85)]['target']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = {'best_log_loss': float(study.best_value), 'best_params': study.best_params}
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BEST_PARAMS_PATH, 'w') as f:
        json.dump(best, f, indent=2)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    best = tune(n_trials=args.trials, timeout=args.timeout)
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
