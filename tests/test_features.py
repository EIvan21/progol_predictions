import sqlite3

import pandas as pd
import pytest

from src.progol import config
from src.progol.features.team_state import build_inference_row
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


def test_market_probs_default_when_missing(synthetic_db):
    # Market probs are built at inference time in team_state, not in
    # calculate_alpha_features. When odds are unavailable the row must fall
    # back to a neutral prior, never zeros/NaN.
    matches = synthetic_db['matches']
    home_id, away_id = int(matches.iloc[0]['home_id']), int(matches.iloc[0]['away_id'])
    conn = sqlite3.connect(synthetic_db['db_path'])
    try:
        row = build_inference_row(conn, home_id, away_id, league_id=39,
                                  date='2024-12-31', market_probs=None)
    finally:
        conn.close()
    assert row['prob_market_h'] > 0
    assert (row['prob_market_h'], row['prob_market_d'], row['prob_market_a']) == (0.45, 0.25, 0.30)


def test_rest_diff_present_and_finite(synthetic_matches):
    out = calculate_alpha_features(synthetic_matches.copy())
    assert 'rest_diff' in out.columns
    assert out['rest_diff'].abs().max() < 1000
