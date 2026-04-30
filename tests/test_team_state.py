import sqlite3

from src.progol import config
from src.progol.features.team_state import build_inference_row


def test_build_inference_row_has_all_features(synthetic_db):
    conn = sqlite3.connect(synthetic_db['db_path'])
    matches = synthetic_db['matches']
    sample = matches.iloc[-5]
    row = build_inference_row(
        conn,
        home_id=int(sample['home_id']), away_id=int(sample['away_id']),
        league_id=int(sample['league_id']), date=sample['date'],
    )
    conn.close()

    expected = set(config.CAT_COLS) | set(config.FEATURE_COLS)
    assert expected.issubset(set(row.keys())), f"missing keys: {expected - set(row.keys())}"


def test_market_probs_default_match_training(synthetic_db):
    """If odds aren't available, inference defaults must equal training fillna."""
    conn = sqlite3.connect(synthetic_db['db_path'])
    sample = synthetic_db['matches'].iloc[-1]
    row = build_inference_row(
        conn,
        home_id=int(sample['home_id']), away_id=int(sample['away_id']),
        league_id=int(sample['league_id']), date=sample['date'],
    )
    conn.close()
    assert row['prob_market_h'] == 0.45
    assert row['prob_market_d'] == 0.25
    assert row['prob_market_a'] == 0.30


def test_elo_diff_changes_with_history(synthetic_db):
    """Confirms Elo is actually computed (not always zero)."""
    conn = sqlite3.connect(synthetic_db['db_path'])
    last = synthetic_db['matches'].iloc[-1]
    row_late = build_inference_row(conn, int(last['home_id']), int(last['away_id']),
                                   int(last['league_id']), last['date'])
    early = synthetic_db['matches'].iloc[5]
    row_early = build_inference_row(conn, int(early['home_id']), int(early['away_id']),
                                    int(early['league_id']), early['date'])
    conn.close()
    assert abs(row_late['elo_diff']) + abs(row_early['elo_diff']) > 0
