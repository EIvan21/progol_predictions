import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_synthetic_matches(n_teams: int = 12, n_matches: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(range(1000, 1000 + n_teams))
    rows = []
    fid = 100000
    base_date = datetime(2024, 1, 1)
    for i in range(n_matches):
        h, a = rng.choice(teams, size=2, replace=False)
        gh = int(rng.poisson(1.4))
        ga = int(rng.poisson(1.2))
        odds_h = round(rng.uniform(1.5, 4.5), 2)
        odds_d = round(rng.uniform(2.8, 4.0), 2)
        odds_a = round(rng.uniform(1.5, 4.5), 2)
        rows.append({
            'fixture_id': fid + i,
            'league_id': 39, 'season': 2024,
            'date': (base_date + timedelta(days=i // 6, hours=int(rng.integers(12, 22)))).isoformat(),
            'venue': f"Stadium_{int(h)}",
            'referee': f"Ref_{int(rng.integers(1, 8))}",
            'home_id': int(h), 'away_id': int(a),
            'goals_home': gh, 'goals_away': ga,
            'status': 'FT',
            'home_shots': int(rng.integers(5, 20)),
            'away_shots': int(rng.integers(5, 20)),
            'home_possession': int(rng.integers(35, 65)),
            'away_possession': int(rng.integers(35, 65)),
            'home_corners': int(rng.integers(0, 10)),
            'away_corners': int(rng.integers(0, 10)),
            'odds_home': odds_h, 'odds_draw': odds_d, 'odds_away': odds_a,
            'odds_movement': 0.0,
            'home_xg': float(round(rng.uniform(0.5, 2.5), 2)),
            'away_xg': float(round(rng.uniform(0.5, 2.5), 2)),
            'home_rank': int(rng.integers(1, 20)), 'away_rank': int(rng.integers(1, 20)),
            'home_form': "WDLWW", 'away_form': "LDWWD",
            'venue_id': int(h), 'venue_surface': "grass",
            'h2h_home_wins': int(rng.integers(0, 5)),
            'h2h_draws': int(rng.integers(0, 3)),
            'h2h_away_wins': int(rng.integers(0, 5)),
            'home_injuries': int(rng.integers(0, 5)),
            'away_injuries': int(rng.integers(0, 5)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    return _make_synthetic_matches()


@pytest.fixture
def synthetic_db(tmp_path, monkeypatch, synthetic_matches):
    """Sets up a temp SQLite DB and patches config paths to point at tmp_path."""
    from src.progol import config as cfg

    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    db_path = data_dir / "progol.db"
    pred_path = data_dir / "predictions.db"

    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", proc_dir, raising=False)
    monkeypatch.setattr(cfg, "MODEL_DIR", model_dir, raising=False)
    monkeypatch.setattr(cfg, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(cfg, "PREDICTIONS_DB_PATH", pred_path, raising=False)
    monkeypatch.setattr(cfg, "TRAIN_CSV", proc_dir / "final_train_data.csv", raising=False)
    monkeypatch.setattr(cfg, "PRIMARY_MODEL_PATH", model_dir / "calibrated_ensemble.pkl", raising=False)
    monkeypatch.setattr(cfg, "METRICS_PATH", model_dir / "metrics.json", raising=False)
    monkeypatch.setattr(cfg, "FEATURE_STATS_PATH", model_dir / "feature_stats.json", raising=False)
    monkeypatch.setattr(cfg, "BEST_PARAMS_PATH", model_dir / "best_params.json", raising=False)
    monkeypatch.setattr(cfg, "LATEST_POINTER_PATH", model_dir / "latest.json", raising=False)

    from src.progol import database
    monkeypatch.setattr(database, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(database, "DATA_DIR", data_dir, raising=False)

    database.init_db()

    conn = sqlite3.connect(db_path)
    synthetic_matches.to_sql("matches", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    return {
        'db_path': db_path,
        'data_dir': data_dir,
        'model_dir': model_dir,
        'matches': synthetic_matches,
    }
