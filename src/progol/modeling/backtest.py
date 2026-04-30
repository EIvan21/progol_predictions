"""Economic backtest: simulate Kelly-staked betting on the holdout vs historical odds.

For each match, computes edge per outcome (model_prob - implied_prob), bets a
fraction of bankroll using fractional Kelly only when edge > min_edge. Reports
final ROI, max drawdown, hit rate, expected vs realized log-loss.

    python -m src.progol.modeling.backtest --kelly 0.25 --min-edge 0.04
"""
import argparse
import json
import logging
import sqlite3
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.progol import config

logger = logging.getLogger(__name__)


def _kelly_fraction(p: float, odds: float) -> float:
    """f* = (p*(odds-1) - (1-p)) / (odds-1). Negative => no bet."""
    if odds <= 1.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    b = odds - 1.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


def simulate(
    probs: np.ndarray,
    targets: np.ndarray,
    odds_h: np.ndarray,
    odds_d: np.ndarray,
    odds_a: np.ndarray,
    kelly_frac: float = 0.25,
    min_edge: float = 0.04,
    bankroll: float = 1000.0,
    max_stake_frac: float = 0.05,
) -> dict:
    cash = bankroll
    history = [cash]
    bets = 0
    wins = 0
    total_stake = 0.0
    total_pnl = 0.0

    for i in range(len(targets)):
        p = probs[i]
        odds = (odds_h[i], odds_d[i], odds_a[i])
        target = int(targets[i])
        for cls in (0, 1, 2):
            implied = 1.0 / odds[cls] if odds[cls] > 0 else 1.0
            edge = float(p[cls]) - implied
            if edge < min_edge:
                continue
            f = _kelly_fraction(float(p[cls]), float(odds[cls])) * kelly_frac
            f = min(f, max_stake_frac)
            if f <= 0:
                continue
            stake = f * cash
            if stake < 1.0:
                continue
            total_stake += stake
            bets += 1
            if cls == target:
                payout = stake * (odds[cls] - 1.0)
                cash += payout
                total_pnl += payout
                wins += 1
            else:
                cash -= stake
                total_pnl -= stake
        history.append(cash)

    arr = np.array(history)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / np.where(peak > 0, peak, 1.0)
    return {
        'starting_bankroll': bankroll,
        'final_bankroll': float(cash),
        'roi_pct': float((cash / bankroll - 1.0) * 100),
        'total_bets': int(bets),
        'wins': int(wins),
        'hit_rate_pct': float(100.0 * wins / bets) if bets else 0.0,
        'total_stake': float(total_stake),
        'total_pnl': float(total_pnl),
        'max_drawdown_pct': float(100.0 * drawdown.max()),
        'kelly_fraction': kelly_frac,
        'min_edge': min_edge,
    }


def _load_holdout() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(config.TRAIN_CSV).sort_values('date').dropna()
    split = int(len(df) * 0.85)
    holdout = df.iloc[split:]
    feature_cols = config.CAT_COLS + config.FEATURE_COLS
    return holdout, holdout[feature_cols], holdout['target'].values


def _odds_for_holdout(holdout: pd.DataFrame) -> tuple:
    conn = sqlite3.connect(config.DB_PATH)
    fids = ','.join(str(int(f)) for f in holdout['fixture_id'].tolist())
    odds = pd.read_sql_query(
        f"SELECT fixture_id, odds_home, odds_draw, odds_away FROM matches WHERE fixture_id IN ({fids})", conn
    )
    conn.close()
    merged = holdout[['fixture_id']].merge(odds, on='fixture_id', how='left')
    return (merged['odds_home'].fillna(0).values,
            merged['odds_draw'].fillna(0).values,
            merged['odds_away'].fillna(0).values)


def run_backtest(kelly: float = 0.25, min_edge: float = 0.04) -> dict:
    if not config.PRIMARY_MODEL_PATH.exists():
        raise FileNotFoundError("Train a model first.")
    pkg = joblib.load(config.PRIMARY_MODEL_PATH)
    model = pkg['model']
    feature_cols = pkg['features']

    holdout, X, y = _load_holdout()
    probs = model.predict_proba(X[feature_cols])
    oh, od, oa = _odds_for_holdout(holdout)

    return simulate(probs, y, oh, od, oa, kelly_frac=kelly, min_edge=min_edge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kelly', type=float, default=0.25)
    parser.add_argument('--min-edge', type=float, default=0.04)
    parser.add_argument('--out', type=str, default=str(config.MODEL_DIR / 'backtest.json'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_backtest(args.kelly, args.min_edge)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
