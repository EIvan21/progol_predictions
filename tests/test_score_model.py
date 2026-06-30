import numpy as np
import pandas as pd
import pytest

from src.progol.modeling import score_model as sm


@pytest.fixture
def intl_df():
    # Synthetic international history: STRONG beats everyone, WEAK loses.
    rng = np.random.default_rng(0)
    teams = ["STRONG", "MID1", "MID2", "WEAK"]
    base_gf = {"STRONG": 2.4, "MID1": 1.3, "MID2": 1.2, "WEAK": 0.6}
    rows, date = [], pd.Timestamp("2020-01-01")
    for i in range(240):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": date + pd.Timedelta(days=i * 3),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(base_gf[h])),
            "away_score": int(rng.poisson(base_gf[a])),
            "neutral": False,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(intl_df):
    return sm.fit_dixon_coles(intl_df, half_life=3650, since="2019-01-01")


@pytest.fixture
def fitted_xgb(intl_df):
    return sm.fit_goal_regressors(intl_df, since="2019-01-01")


def test_score_matrix_is_a_distribution(fitted_model):
    M = sm.score_matrix(fitted_model, "STRONG", "WEAK", neutral=True)
    assert M.shape == (7, 7)
    assert M.min() >= 0
    assert M.sum() == pytest.approx(1.0)


def test_outcome_probs_sum_to_one(fitted_model):
    M = sm.score_matrix(fitted_model, "MID1", "MID2")
    pH, pD, pA = sm.outcome_probs(M)
    assert pH + pD + pA == pytest.approx(1.0)


def test_favourite_beats_underdog(fitted_model):
    M = sm.score_matrix(fitted_model, "STRONG", "WEAK", neutral=True)
    pH, _, pA = sm.outcome_probs(M)
    assert pH > pA


def test_top_scores_sorted_and_capped(fitted_model):
    M = sm.score_matrix(fitted_model, "STRONG", "WEAK")
    top = sm.top_scores(M, 5)
    assert len(top) == 5
    assert list(top["prob"]) == sorted(top["prob"], reverse=True)


def test_resolve_team_uses_name_map(fitted_model):
    # Exact + fuzzy fall through to None for genuine unknowns.
    assert sm.resolve_team("STRONG", fitted_model) == "STRONG"
    assert sm.resolve_team("Atlantis", fitted_model) is None


def test_xgb_matrix_is_a_distribution(fitted_xgb):
    M = sm.score_matrix_xgb(fitted_xgb, "STRONG", "WEAK", neutral=True)
    assert M.shape == (7, 7)
    assert M.min() >= 0
    assert M.sum() == pytest.approx(1.0)


def test_xgb_favourite_beats_underdog(fitted_xgb):
    M = sm.score_matrix_xgb(fitted_xgb, "STRONG", "WEAK", neutral=True)
    pH, _, pA = sm.outcome_probs(M)
    assert pH > pA


def test_xgb_save_load_roundtrip(fitted_xgb, tmp_path):
    path = tmp_path / "score_model.pkl"
    sm.save_model(fitted_xgb, path)
    loaded = sm.load_model(path)
    a = sm.score_matrix_xgb(fitted_xgb, "STRONG", "WEAK")
    b = sm.score_matrix_xgb(loaded, "STRONG", "WEAK")
    assert np.allclose(a, b)


def test_evaluate_reports_both_backends(intl_df):
    res = sm.evaluate(intl_df, test_since="2021-06-01", since="2019-01-01")
    assert res["n_test"] > 0
    for tag in ("dc", "xgb"):
        assert 0.0 <= res[tag]["accuracy"] <= 1.0
        assert res[tag]["log_loss"] > 0


def test_score_eval_appends_history(intl_df, tmp_path):
    from src.progol.reporting import score_eval
    path = tmp_path / "score_model_history.csv"
    r1 = score_eval.run(df=intl_df, test_since="2021-06-01", history_path=path)
    assert path.exists() and r1["n_test"] > 0
    score_eval.run(df=intl_df, test_since="2021-06-01", history_path=path)
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("date,")  # header once
    assert len(lines) == 3              # header + two appended rows
