import pandas as pd
import pytest

from src.progol import config
from src.progol.modeling.preprocess import calculate_alpha_features


def test_alpha_features_columns_match_config(synthetic_matches):
    out = calculate_alpha_features(synthetic_matches.copy())
    expected = {'fixture_id', 'date', 'target'} | set(config.CAT_COLS) | set(config.FEATURE_COLS)
    assert expected.issubset(set(out.columns)), f"missing: {expected - set(out.columns)}"


def test_target_values_in_range(synthetic_matches):
    out = calculate_alpha_features(synthetic_matches.copy())
    assert set(out['target'].unique()).issubset({0, 1, 2})


def test_no_inf_or_nan(synthetic_matches):
    out = calculate_alpha_features(synthetic_matches.copy())
    for col in config.FEATURE_COLS:
        assert out[col].isna().sum() == 0, f"{col} has NaN"
        assert (out[col].abs() != float('inf')).all(), f"{col} has inf"


def test_market_probs_default_when_missing(synthetic_matches):
    df = synthetic_matches.copy()
    df.loc[df.index[:50], ['odds_home', 'odds_draw', 'odds_away']] = None
    out = calculate_alpha_features(df)
    assert (out['prob_market_h'] > 0).all()


def test_rest_diff_present_and_finite(synthetic_matches):
    out = calculate_alpha_features(synthetic_matches.copy())
    assert 'rest_diff' in out.columns
    assert out['rest_diff'].abs().max() < 1000
