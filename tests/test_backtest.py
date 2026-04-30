import numpy as np

from src.progol.modeling.backtest import _kelly_fraction, simulate


def test_kelly_no_bet_on_negative_edge():
    assert _kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_positive_edge():
    f = _kelly_fraction(0.6, 2.0)
    assert 0 < f < 1


def test_simulate_runs_and_returns_metrics():
    rng = np.random.default_rng(0)
    n = 100
    probs = rng.dirichlet([2, 1, 2], size=n)
    targets = np.argmax(probs, axis=1)
    odds_h = np.full(n, 2.0)
    odds_d = np.full(n, 3.5)
    odds_a = np.full(n, 2.5)
    res = simulate(probs, targets, odds_h, odds_d, odds_a, kelly_frac=0.25, min_edge=0.05)
    assert 'roi_pct' in res
    assert 'final_bankroll' in res
    assert res['final_bankroll'] >= 0
