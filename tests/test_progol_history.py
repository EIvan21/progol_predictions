"""Tests for the Progol-history feature: database.get_concurso_hits +
reporting.progol_history.summarize_recent + the recap formatter.

These guard the user-visible "we got 8/14 last week" pipeline against
silent breakage when the progol_concurso_games schema or the
predicted/actual semantics change.
"""
import sqlite3

import pytest

from src.progol import database
from src.progol.bot.formatting import format_previous_concurso_recap
from src.progol.reporting import progol_history


# --- fixture ---------------------------------------------------------------

@pytest.fixture
def concurso_db(tmp_path, monkeypatch):
    """Temp SQLite DB with the concurso schema initialized. Each test
    seeds its own rows so assertions stay local to the test."""
    db_path = tmp_path / "progol.db"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    from src.progol import config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(cfg, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(database, "DB_PATH", db_path, raising=False)
    monkeypatch.setattr(database, "DATA_DIR", data_dir, raising=False)

    database.init_db()
    return db_path


def _seed_concurso(db_path, concurso_number, rows):
    """`rows` is a list of (game_number, predicted_label, actual_label).
    None means "missing" (NULL in DB)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO progol_concursos (concurso_number, week_start, source_url, scraped_at) "
        "VALUES (?, '2026-05-15', 'test', datetime('now'))",
        (concurso_number,),
    )
    for gn, pred, actual in rows:
        conn.execute(
            "INSERT OR REPLACE INTO progol_concurso_games "
            "(concurso_number, game_number, home_name, away_name, fixture_id, "
            " predicted_label, actual_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (concurso_number, gn, f"H{gn}", f"A{gn}", 10000 + gn, pred, actual),
        )
    conn.commit()
    conn.close()


# --- get_concurso_hits -----------------------------------------------------

def test_get_concurso_hits_splits_main_and_revancha(concurso_db):
    """game_number 1..14 → main, 15..21 → revancha. Both buckets populate
    independently."""
    rows = [(gn, 0, 0) for gn in range(1, 22)]  # all predicted L, all actual L → 21 hits
    _seed_concurso(concurso_db, 2400, rows)
    h = database.get_concurso_hits(2400)
    assert h['concurso_number'] == 2400
    assert h['main']['total'] == 14
    assert h['main']['hits'] == 14
    assert h['main']['compared'] == 14
    assert h['revancha']['total'] == 7
    assert h['revancha']['hits'] == 7
    assert h['revancha']['compared'] == 7


def test_get_concurso_hits_counts_only_correct_matches_as_hits(concurso_db):
    """Hit = predicted_label == actual_label. Different labels = miss."""
    rows = [
        (1, 0, 0),  # hit
        (2, 0, 1),  # miss
        (3, 1, 1),  # hit
        (4, 2, 0),  # miss
    ]
    _seed_concurso(concurso_db, 2401, rows)
    h = database.get_concurso_hits(2401)
    assert h['main']['hits'] == 2
    assert h['main']['compared'] == 4


def test_get_concurso_hits_compared_vs_settled(concurso_db):
    """`compared` requires BOTH labels present. `settled` only requires
    actual_label. Used to distinguish "we whiffed everything" from
    "we never predicted" (the cn 2326-2330 case in real data)."""
    rows = [
        (1, 0, 0),       # both → compared & settled, hit
        (2, None, 1),    # actual only → settled but NOT compared
        (3, 1, None),    # pred only → neither
        (4, None, None), # neither
    ]
    _seed_concurso(concurso_db, 2402, rows)
    h = database.get_concurso_hits(2402)
    m = h['main']
    assert m['total'] == 4
    assert m['predicted'] == 2
    assert m['settled'] == 2
    assert m['compared'] == 1
    assert m['hits'] == 1


def test_get_concurso_hits_slots_carry_correct_flag(concurso_db):
    """Per-slot list lets future diagnostics show WHICH slots we missed.
    `correct` is True/False/None matching get_concurso_hits semantics."""
    rows = [
        (1, 0, 0),       # correct True
        (2, 0, 2),       # correct False
        (3, None, 0),    # correct None (no prediction)
        (4, 1, None),    # correct None (no actual)
    ]
    _seed_concurso(concurso_db, 2403, rows)
    h = database.get_concurso_hits(2403)
    slots = {s['game_number']: s['correct'] for s in h['main']['slots']}
    assert slots == {1: True, 2: False, 3: None, 4: None}


def test_get_concurso_hits_unknown_concurso_returns_empty_buckets(concurso_db):
    """Querying a concurso that doesn't exist must not crash."""
    h = database.get_concurso_hits(99999)
    assert h['main']['total'] == 0
    assert h['main']['slots'] == []
    assert h['revancha']['total'] == 0


# --- get_previous_concurso_number -----------------------------------------

def test_get_previous_concurso_number_returns_max_below(concurso_db):
    """Gaps don't break this — get the LARGEST existing concurso strictly
    less than the input."""
    for cn in [2300, 2305, 2310]:
        _seed_concurso(concurso_db, cn, [(1, 0, 0)])
    assert database.get_previous_concurso_number(2310) == 2305
    assert database.get_previous_concurso_number(2305) == 2300
    assert database.get_previous_concurso_number(2300) is None
    # Non-existent reference still gets the max below it.
    assert database.get_previous_concurso_number(2400) == 2310


# --- list_recent_concursos ------------------------------------------------

def test_list_recent_concursos_orders_descending(concurso_db):
    for cn in [2200, 2201, 2202, 2203]:
        _seed_concurso(concurso_db, cn, [(1, 0, 0)])
    assert database.list_recent_concursos(n=3) == [2203, 2202, 2201]
    assert database.list_recent_concursos(n=10) == [2203, 2202, 2201, 2200]


# --- summarize_recent + status classification -----------------------------

def test_summarize_recent_status_complete(concurso_db):
    """All 14 main slots predicted AND compared → complete."""
    rows = [(gn, 0, 0) for gn in range(1, 15)] + [(gn, 1, 1) for gn in range(15, 22)]
    _seed_concurso(concurso_db, 2500, rows)
    out = progol_history.summarize_recent(n=5)
    assert len(out) == 1
    assert out[0]['status'] == 'complete'
    assert out[0]['main']['hits'] == 14
    assert out[0]['revancha']['hits'] == 7


def test_summarize_recent_status_in_progress(concurso_db):
    """Predicted all 14 but only some have actuals → in_progress."""
    rows = [(gn, 0, 0 if gn <= 7 else None) for gn in range(1, 15)]
    _seed_concurso(concurso_db, 2501, rows)
    out = progol_history.summarize_recent(n=5)
    assert out[0]['status'] == 'in_progress'
    assert out[0]['main']['compared'] == 7
    assert out[0]['main']['predicted'] == 14


def test_summarize_recent_status_no_predictions(concurso_db):
    """Actuals only, no predicted labels → no_predictions (the real
    2326-2330 case where the bot booted after the concurso closed)."""
    rows = [(gn, None, 0) for gn in range(1, 15)]
    _seed_concurso(concurso_db, 2502, rows)
    out = progol_history.summarize_recent(n=5)
    assert out[0]['status'] == 'no_predictions'
    assert out[0]['main']['compared'] == 0


def test_summarize_recent_orders_newest_first(concurso_db):
    for cn in [2100, 2101, 2102]:
        _seed_concurso(concurso_db, cn, [(1, 0, 0)])
    out = progol_history.summarize_recent(n=10)
    assert [s['concurso_number'] for s in out] == [2102, 2101, 2100]


# --- score formatter ------------------------------------------------------

def test_fmt_score_no_predictions(concurso_db):
    """compared=0 (regardless of why) → '-/{total}'."""
    bucket = {'total': 14, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0}
    assert progol_history._fmt_score(bucket) == '-/14'


def test_fmt_score_partial(concurso_db):
    bucket = {'total': 14, 'predicted': 14, 'settled': 8, 'compared': 8, 'hits': 5}
    assert progol_history._fmt_score(bucket) == '5/8 of 14'


def test_fmt_score_complete(concurso_db):
    bucket = {'total': 14, 'predicted': 14, 'settled': 14, 'compared': 14, 'hits': 9}
    assert progol_history._fmt_score(bucket) == '9/14'


def test_fmt_score_em_dash_for_html(concurso_db):
    """HTML callers can pass an em-dash; helper must respect it."""
    bucket = {'total': 7, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0}
    assert progol_history._fmt_score(bucket, dash='—') == '—/7'


# --- recap formatter (Telegram send_predictions append) -------------------

def test_recap_returns_none_when_nothing_comparable():
    """Both buckets compared=0 → no recap. Prevents an empty/misleading
    "Concurso pasado #N: " line in the Telegram message."""
    hits = {
        'main': {'total': 14, 'predicted': 0, 'settled': 12, 'compared': 0, 'hits': 0},
        'revancha': {'total': 7, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0},
    }
    assert format_previous_concurso_recap(2330, hits) is None


def test_recap_with_main_only():
    """Revancha not comparable; still emits a recap with just the main line."""
    hits = {
        'main': {'total': 14, 'predicted': 14, 'settled': 14, 'compared': 14, 'hits': 8},
        'revancha': {'total': 7, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0},
    }
    recap = format_previous_concurso_recap(2331, hits)
    assert recap is not None
    assert '#2331' in recap
    assert 'Main 8/14' in recap
    assert 'Rev' not in recap


def test_recap_emoji_thresholds():
    """Display-only thresholds: >=55% ✅, >=40% ➖, <40% ❌. Sanity-checks
    so a future tweak doesn't silently shift the cutoffs."""
    def _make(hits_n):
        return {
            'main': {'total': 14, 'predicted': 14, 'settled': 14, 'compared': 14, 'hits': hits_n},
            'revancha': {'total': 7, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0},
        }
    assert '✅' in format_previous_concurso_recap(1, _make(9))   # 64% → ✅
    assert '➖' in format_previous_concurso_recap(1, _make(7))   # 50% — between cutoffs → wait, 7/14=50% is < 55 and >= 40 → ➖
    assert '❌' in format_previous_concurso_recap(1, _make(4))   # 28% → ❌


def test_recap_handles_partial_with_suffix():
    """Compared < total → recap shows the "of N" suffix so the reader
    knows the rate is on a partial sample."""
    hits = {
        'main': {'total': 14, 'predicted': 14, 'settled': 9, 'compared': 9, 'hits': 6},
        'revancha': {'total': 7, 'predicted': 0, 'settled': 0, 'compared': 0, 'hits': 0},
    }
    recap = format_previous_concurso_recap(2332, hits)
    assert '6/9 of 14' in recap


def test_recap_handles_none_input():
    """Pure defensiveness — caller's DB lookup might return falsy."""
    assert format_previous_concurso_recap(2330, None) is None
    assert format_previous_concurso_recap(2330, {}) is None
