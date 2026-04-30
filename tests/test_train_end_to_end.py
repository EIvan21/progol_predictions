import json

import joblib
import pytest

from src.progol import config
from src.progol.modeling.preprocess import calculate_alpha_features


@pytest.mark.slow
def test_train_minimal_end_to_end(synthetic_db):
    """Trains on synthetic data, asserts model saves with versioning + valid probs."""
    processed = calculate_alpha_features(synthetic_db['matches'].copy())
    processed.to_csv(config.TRAIN_CSV, index=False)

    from src.progol.modeling import train as train_mod
    train_mod.train_heavy_model()

    assert config.PRIMARY_MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()
    assert config.LATEST_POINTER_PATH.exists()
    assert config.FEATURE_STATS_PATH.exists()

    pkg = joblib.load(config.PRIMARY_MODEL_PATH)
    assert 'model' in pkg and 'features' in pkg and 'version' in pkg

    pointer = json.loads(config.LATEST_POINTER_PATH.read_text())
    assert pointer['version'].startswith('v_')

    metrics = json.loads(config.METRICS_PATH.read_text())
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert metrics['log_loss'] > 0

    X = processed.iloc[-5:][config.CAT_COLS + config.FEATURE_COLS]
    probs = pkg['model'].predict_proba(X)
    assert probs.shape == (5, 3)
    assert (probs >= 0).all() and (probs <= 1).all()
    row_sums = probs.sum(axis=1)
    assert ((row_sums > 0.99) & (row_sums < 1.01)).all()
