"""Dixon-Coles bivariate Poisson 1X2 estimator.

Why this exists in the stack: LightGBM and CatBoost both learn 1X2 from
tabular features by splaying decision boundaries — same inductive bias.
Poisson DC models the football-specific generative process directly
(goals are Poisson counts; low-scoring scorelines are over-represented
vs independence). That gives the stacking meta-learner a genuinely
different view to ensemble against, not just another flavor of GBM.

Limitations vs full Dixon-Coles MLE:
- Doesn't learn per-team attack/defense from goals data (would need
  ~1342 × 2 params + heavy regularization for our 60k row dataset)
- Uses pre-computed EWMA team strengths as a proxy for attack/defense
- dc_rho fixed at -0.1 (literature default for top European leagues)
- home_advantage fixed at exp(0.18) ≈ 1.20× scoring multiplier

Reference: Dixon & Coles 1997, "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from sklearn.base import BaseEstimator, ClassifierMixin


# EWMA columns the estimator reads from X. Names match what preprocess.py
# generates via features/rolling.py. League id is optional — if absent,
# the global baseline is used for all rows.
EWMA_COLS = {
    'home_gf': 'home_gf_ewma',
    'away_gf': 'away_gf_ewma',
    'home_ga': 'home_ga_ewma',
    'away_ga': 'away_ga_ewma',
}


class PoissonDCEstimator(BaseEstimator, ClassifierMixin):
    """sklearn-compatible 1X2 classifier with Dixon-Coles low-score correction.

    Inputs (columns expected in X):
        home_gf_ewma, away_gf_ewma, home_ga_ewma, away_ga_ewma
        league_id (optional — enables per-league lambda baselines)

    fit() computes the global average gf/ga and per-league baselines.
    predict_proba(X) builds a (max_goals+1)² Poisson score grid per row,
    applies the DC tau correction to the four low-scoring cells, and
    sums tril / diag / triu to [P(home), P(draw), P(away)].
    """

    def __init__(self, dc_rho: float = -0.1, max_goals: int = 8,
                 home_adv_logit: float = 0.18, eps: float = 0.5,
                 lambda_cap: float = 6.0):
        self.dc_rho = dc_rho
        self.max_goals = max_goals
        self.home_adv_logit = home_adv_logit
        self.eps = eps
        self.lambda_cap = lambda_cap

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1, 2])

        # EWMA can be 0 for cold-start teams; floor at `eps` to avoid
        # 0-lambda collapsing the Poisson to a delta at 0 goals.
        h_gf = np.maximum(X[EWMA_COLS['home_gf']].values, self.eps)
        h_ga = np.maximum(X[EWMA_COLS['home_ga']].values, self.eps)
        a_gf = np.maximum(X[EWMA_COLS['away_gf']].values, self.eps)
        a_ga = np.maximum(X[EWMA_COLS['away_ga']].values, self.eps)

        self.global_avg_gf_ = float(np.mean(h_gf))
        self.global_avg_ga_ = float(np.mean(h_ga))

        # Per-league average goals — mean of (home_gf + away_gf) EWMA within
        # the league. Used to scale lambdas to league-specific scoring
        # rates (Bundesliga ~3.0 goals/match vs Liga MX ~2.5).
        self.league_avg_ = {}
        if 'league_id' in X.columns:
            for lid, group in X.groupby('league_id'):
                if len(group) == 0:
                    continue
                gf1 = np.maximum(group[EWMA_COLS['home_gf']].values, self.eps).mean()
                gf2 = np.maximum(group[EWMA_COLS['away_gf']].values, self.eps).mean()
                self.league_avg_[int(lid)] = float((gf1 + gf2) / 2)

        return self

    def predict_proba(self, X):
        n = len(X)
        h_gf = np.maximum(X[EWMA_COLS['home_gf']].values, self.eps)
        h_ga = np.maximum(X[EWMA_COLS['home_ga']].values, self.eps)
        a_gf = np.maximum(X[EWMA_COLS['away_gf']].values, self.eps)
        a_ga = np.maximum(X[EWMA_COLS['away_ga']].values, self.eps)
        lids = (X['league_id'].values if 'league_id' in X.columns
                else np.zeros(n, dtype=int))

        home_adv = float(np.exp(self.home_adv_logit))
        probs = np.zeros((n, 3))
        for i in range(n):
            baseline = self.league_avg_.get(int(lids[i]), self.global_avg_gf_)
            h_attack = h_gf[i] / self.global_avg_gf_
            a_weakness = a_ga[i] / self.global_avg_ga_
            a_attack = a_gf[i] / self.global_avg_gf_
            h_weakness = h_ga[i] / self.global_avg_ga_
            lam_h = min(baseline * h_attack * a_weakness * home_adv,
                        self.lambda_cap)
            lam_a = min(baseline * a_attack * h_weakness / home_adv,
                        self.lambda_cap)
            probs[i] = self._dc_probs(lam_h, lam_a)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _dc_probs(self, lam_h: float, lam_a: float) -> np.ndarray:
        max_g = self.max_goals
        p_h = poisson.pmf(np.arange(max_g + 1), lam_h)
        p_a = poisson.pmf(np.arange(max_g + 1), lam_a)
        grid = np.outer(p_h, p_a)

        # Dixon-Coles tau correction for (0,0), (0,1), (1,0), (1,1).
        # Negative rho boosts (0,0) and (1,1), suppresses (0,1) and (1,0)
        # — matches the empirical excess of draws + scoreless ties.
        rho = self.dc_rho
        tau = np.array([
            [1.0 - lam_h * lam_a * rho, 1.0 + lam_h * rho],
            [1.0 + lam_a * rho, 1.0 - rho],
        ])
        grid[:2, :2] *= tau

        # Renormalize because tau adjustments shift total mass.
        grid /= grid.sum()

        # Sum to [P(home win), P(draw), P(away win)].
        p_home_win = float(np.tril(grid, -1).sum())
        p_draw = float(np.diag(grid).sum())
        p_away_win = float(np.triu(grid, 1).sum())

        probs = np.array([p_home_win, p_draw, p_away_win])
        probs = np.clip(probs, 1e-9, 1.0)
        return probs / probs.sum()
