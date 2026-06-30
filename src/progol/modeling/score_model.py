"""Exact-score models for national-team fixtures. Two backends:

1. Dixon-Coles (`fit_dixon_coles` / `score_matrix`): time-weighted
   attack/defence ratings (+ home advantage, low-score correlation `rho`) fit by
   maximum likelihood. Fast, interpretable, no training artifact.
2. XGBoost goal regressors (`fit_goal_regressors` / `score_matrix_xgb`): genuine
   ML — two `count:poisson` regressors predict each team's expected goals from
   Dixon-Coles strengths + recent form, then Poisson builds the matrix. Trains
   and retrains on accumulating results; persist with `save_model`/`load_model`.

Both produce the same score-matrix shape, so `outcome_probs` / `top_scores` and
the report layer work with either.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from thefuzz import fuzz, process

logger = logging.getLogger(__name__)

# Slate uses API-Football English names; martj42 spells a few differently.
NAME_MAP = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

_DEFAULT_SINCE = "2014-01-01"
_DEFAULT_HALF_LIFE = 730  # days; ~2-year memory for current squad strength


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles dependence correction for the four low-scoring cells."""
    t = np.ones_like(lam, dtype=float)
    a = (x == 0) & (y == 0); t[a] = 1 - lam[a] * mu[a] * rho
    a = (x == 0) & (y == 1); t[a] = 1 + lam[a] * rho
    a = (x == 1) & (y == 0); t[a] = 1 + mu[a] * rho
    a = (x == 1) & (y == 1); t[a] = 1 - rho
    return t


def fit_dixon_coles(df: pd.DataFrame, cutoff=None, since: str = _DEFAULT_SINCE,
                    half_life: int = _DEFAULT_HALF_LIFE) -> dict:
    """Fit on played matches before `cutoff`. `df` needs columns date,
    home_team, away_team, home_score, away_score, neutral."""
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else df["date"].max() + pd.Timedelta(days=1)
    train = df[(df["date"] >= pd.Timestamp(since)) & (df["date"] < cutoff)].copy()
    train["w"] = 0.5 ** ((cutoff - train["date"]).dt.days / half_life)

    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train["home_team"].map(idx).values
    ai = train["away_team"].map(idx).values
    hs = train["home_score"].values
    as_ = train["away_score"].values
    w = train["w"].values
    # Home advantage only applies to non-neutral venues.
    lm = (~train["neutral"].astype(bool)).values.astype(float)

    def nll(p):
        at = p[:n] - p[:n].mean()
        dn = p[n:2 * n]
        h, r = p[2 * n], p[2 * n + 1]
        lam = np.exp(at[hi] - dn[ai] + h * lm)
        mu = np.exp(at[ai] - dn[hi])
        t = np.clip(_tau(hs, as_, lam, mu, r), 1e-10, None)
        return -(w * (np.log(t) + poisson.logpmf(hs, lam) + poisson.logpmf(as_, mu))).sum()

    init = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
    res = minimize(nll, init, method="L-BFGS-B")
    att = res.x[:n] - res.x[:n].mean()
    logger.info("fit dixon-coles on %d matches, %d teams", len(train), n)
    return {"att": att, "dfn": res.x[n:2 * n], "home": res.x[2 * n],
            "rho": res.x[2 * n + 1], "idx": idx, "teams": teams}


def resolve_team(name: str, model: dict, threshold: int = 85):
    """Map a slate team name to a name the model knows, or None."""
    known = model["idx"]
    if name in known:
        return name
    mapped = NAME_MAP.get(name)
    if mapped and mapped in known:
        return mapped
    cand, sc = process.extractOne(name, list(known), scorer=fuzz.token_sort_ratio)
    if sc >= threshold:
        return cand
    logger.warning("unresolved score-model team '%s' (best '%s' @ %d)", name, cand, sc)
    return None


def score_matrix(model: dict, home: str, away: str, neutral: bool = False,
                 max_goals: int = 7) -> np.ndarray:
    """Normalised P(home_goals, away_goals) matrix, shape (max_goals, max_goals)."""
    h, a = model["idx"][home], model["idx"][away]
    att, dfn, rho = model["att"], model["dfn"], model["rho"]
    host = 0.0 if neutral else 1.0
    lam = np.exp(att[h] - dfn[a] + model["home"] * host)
    mu = np.exp(att[a] - dfn[h])
    M = np.outer(poisson.pmf(np.arange(max_goals), lam),
                 poisson.pmf(np.arange(max_goals), mu))
    # Dixon-Coles low-score correction on the four cells.
    M[0, 0] *= 1 - lam * mu * rho
    M[0, 1] *= 1 + lam * rho
    M[1, 0] *= 1 + mu * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0.0, None)
    return M / M.sum()


def outcome_probs(M: np.ndarray):
    """Return (P_home_win, P_draw, P_away_win) from a score matrix."""
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def top_scores(M: np.ndarray, n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            result = "home" if i > j else ("draw" if i == j else "away")
            rows.append({"score": f"{i}-{j}", "prob": M[i, j] * 100, "result": result})
    return pd.DataFrame(rows).sort_values("prob", ascending=False).head(n).reset_index(drop=True)


# --- XGBoost goal-regressor backend (genuine ML) --------------------------
# Two count:poisson regressors predict each team's expected goals from the
# Dixon-Coles strengths + recent form; Poisson then builds the score matrix.
# Trains/retrains on accumulating results — the more matches settle, the more
# the regressors learn.

def _long_form(df: pd.DataFrame, since: str):
    """Per-team rolling (last-5) points/goals form, plus each team's latest
    available form (for predicting future fixtures)."""
    sub = df[df["date"] >= pd.Timestamp(since)]
    rows = []
    for r in sub.itertuples():
        rows.append((r.date, r.home_team, r.home_score, r.away_score, 1))
        rows.append((r.date, r.away_team, r.away_score, r.home_score, 0))
    L = pd.DataFrame(rows, columns=["date", "team", "gf", "ga", "ishome"]).sort_values("date")
    L["pts"] = np.where(L.gf > L.ga, 3, np.where(L.gf == L.ga, 1, 0))
    g = L.groupby("team")
    L["form5"] = g["pts"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    L["gf5"] = g["gf"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    L["ga5"] = g["ga"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    by_row = {(r.date, r.team, r.ishome): (r.form5, r.gf5, r.ga5) for r in L.itertuples()}
    latest = {}
    for r in L.itertuples():  # sorted by date, so last write per (team,ishome) wins
        if not (pd.isna(r.form5) or pd.isna(r.gf5) or pd.isna(r.ga5)):
            latest[(r.team, r.ishome)] = (r.form5, r.gf5, r.ga5)
    return by_row, latest


def _feature_row(dc: dict, home: str, away: str, host: float, fh, fa):
    att, dfn = dc["att"], dc["dfn"]
    ih, ia = dc["idx"][home], dc["idx"][away]
    lam = np.exp(att[ih] - dfn[ia] + dc["home"] * host)
    mu = np.exp(att[ia] - dfn[ih])
    return [lam, mu, att[ih] - att[ia], dfn[ih] - dfn[ia], float(host),
            fh[0], fh[1], fh[2], fa[0], fa[1], fa[2]]


def fit_goal_regressors(df: pd.DataFrame, dc: dict = None, cutoff=None,
                        since: str = _DEFAULT_SINCE) -> dict:
    """Train two XGBoost count:poisson regressors (home/away goals) on matches
    before `cutoff`. Returns a model dict consumable by `score_matrix_xgb`."""
    import xgboost as xgb

    cutoff = pd.Timestamp(cutoff) if cutoff is not None else df["date"].max() + pd.Timedelta(days=1)
    train = df[(df["date"] >= pd.Timestamp(since)) & (df["date"] < cutoff)]
    if dc is None:
        dc = fit_dixon_coles(df, cutoff=cutoff, since=since)
    by_row, latest = _long_form(train, since)
    known = dc["idx"]

    X, yh, ya = [], [], []
    for r in train.itertuples():
        if r.home_team not in known or r.away_team not in known:
            continue
        host = 0.0 if r.neutral else 1.0
        fh = by_row.get((r.date, r.home_team, 1), (1.0, 1.0, 1.0))
        fa = by_row.get((r.date, r.away_team, 0), (1.0, 1.0, 1.0))
        feat = _feature_row(dc, r.home_team, r.away_team, host, fh, fa)
        if any(pd.isna(feat)):
            continue
        X.append(feat); yh.append(r.home_score); ya.append(r.away_score)
    X = np.array(X); yh = np.array(yh); ya = np.array(ya)

    params = dict(objective="count:poisson", n_estimators=300, max_depth=4,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    reg_home = xgb.XGBRegressor(**params).fit(X, yh)
    reg_away = xgb.XGBRegressor(**params).fit(X, ya)
    logger.info("fit xgb goal regressors on %d matches", len(X))
    return {"reg_home": reg_home, "reg_away": reg_away, "dc": dc,
            "latest_form": latest, "n_train": int(len(X))}


def score_matrix_xgb(model: dict, home: str, away: str, neutral: bool = False,
                     max_goals: int = 7) -> np.ndarray:
    dc = model["dc"]
    host = 0.0 if neutral else 1.0
    fh = model["latest_form"].get((home, 1), (1.0, 1.0, 1.0))
    fa = model["latest_form"].get((away, 0), (1.0, 1.0, 1.0))
    feat = np.array(_feature_row(dc, home, away, host, fh, fa), dtype=float).reshape(1, -1)
    lam = float(model["reg_home"].predict(feat)[0])
    mu = float(model["reg_away"].predict(feat)[0])
    M = np.outer(poisson.pmf(np.arange(max_goals), lam),
                 poisson.pmf(np.arange(max_goals), mu))
    return M / M.sum()


def resolve_team_xgb(name: str, model: dict, threshold: int = 85):
    return resolve_team(name, model["dc"], threshold)


def evaluate(df: pd.DataFrame, test_since: str, since: str = _DEFAULT_SINCE) -> dict:
    """Temporal holdout: train both backends on matches in [since, test_since),
    score the 1X2 outcome of every match on/after test_since, and report
    accuracy + log-loss per backend. This is the metric to track over time as
    the models retrain on accumulating results."""
    cutoff = pd.Timestamp(test_since)
    dc = fit_dixon_coles(df, cutoff=cutoff, since=since)
    xgb = fit_goal_regressors(df, dc=dc, cutoff=cutoff, since=since)
    known = dc["idx"]
    test = df[(df["date"] >= cutoff) & df["home_team"].isin(known) & df["away_team"].isin(known)]

    stats = {"dc": [0, 0.0], "xgb": [0, 0.0]}  # [correct, logloss_sum]
    n = 0
    for r in test.itertuples():
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        for tag, M in (("dc", score_matrix(dc, r.home_team, r.away_team, neutral=bool(r.neutral))),
                       ("xgb", score_matrix_xgb(xgb, r.home_team, r.away_team, neutral=bool(r.neutral)))):
            probs = list(outcome_probs(M))
            if int(np.argmax(probs)) == actual:
                stats[tag][0] += 1
            stats[tag][1] += -np.log(max(probs[actual], 1e-9))
        n += 1
    out = {"n_test": n}
    for tag in ("dc", "xgb"):
        out[tag] = {"accuracy": stats[tag][0] / n if n else 0.0,
                    "log_loss": stats[tag][1] / n if n else 0.0}
    return out


def save_model(model: dict, path) -> None:
    import joblib
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path) -> dict:
    import joblib
    return joblib.load(path)

