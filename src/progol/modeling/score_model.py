"""Dixon-Coles exact-score model for national-team fixtures.

Fits time-weighted attack/defence ratings (+ home advantage and the Dixon-Coles
low-score correlation `rho`) on international history, then builds a per-fixture
score matrix P(home_goals, away_goals). The 1X2 and top-scoreline views are
derived from that matrix. Kept backend-agnostic: a future MCMC backend can
return the same `score_matrix` shape.
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
