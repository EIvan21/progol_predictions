import json
import sqlite3

import pytest

from src.progol import database
from src.progol.reporting import performance as perf


@pytest.fixture
def settled_db(tmp_path, monkeypatch):
    """Temp DB with one fully-settled concurso: model always predicts L; odd
    games end L (hit), even games end E (miss, and never called)."""
    db = tmp_path / "progol.db"
    monkeypatch.setattr(database, "DB_PATH", db, raising=False)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE progol_concurso_games (
        concurso_number INTEGER, game_number INTEGER, home_name TEXT, away_name TEXT,
        fixture_id INTEGER, predicted_label INTEGER, predicted_probs TEXT,
        actual_label INTEGER, settled_at TEXT)""")
    probs = json.dumps({"L": 0.5, "E": 0.25, "V": 0.25})
    for g in range(1, 22):
        actual = 0 if g % 2 else 1  # odd -> L, even -> E
        conn.execute("INSERT INTO progol_concurso_games VALUES (?,?,?,?,?,?,?,?,?)",
                     (100, g, "Home", "Away", 1000 + g, 0, probs, actual, "now"))
    conn.commit(); conn.close()
    return db


def test_evaluate_splits_main_and_revancha(settled_db):
    r = perf.evaluate(n=1)
    assert r["n_concursos"] == 1
    m, rv = r["main"], r["revancha"]
    assert m["total_comp"] == 14 and m["total_hits"] == 7   # 7 odd games hit
    assert rv["total_comp"] == 7 and rv["total_hits"] == 4
    # objective metric present
    assert m["expected_hits"] == pytest.approx(14 * 0.5)


def test_never_predicts_draws_is_measured(settled_db):
    r = perf.evaluate(n=1)
    m = r["main"]
    assert m["draws_real"] == 7 and m["draws_hit"] == 0


def test_render_telegram_cites_source(settled_db):
    msg = perf.render_telegram(n=1)
    assert "API-Football" in msg
    assert "PRINCIPALES" in msg and "REVANCHA" in msg


def test_reason_flags_uncalled_draw():
    assert perf._reason(0, {"L": 0.5, "E": 0.25, "V": 0.25}, 1) == "empate no llamado"
