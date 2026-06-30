"""National-team match history from the public martj42/international_results
dataset. progol.db only holds club leagues (fetched for the 1X2 model), so the
exact-score model fits on this CSV for national-team fixtures."""
import logging
import time

import pandas as pd
import requests

from src.progol import config

logger = logging.getLogger(__name__)

# Re-download at most once a day; the upstream CSV updates after match days.
_MAX_CACHE_AGE_S = 24 * 3600


def _cache_is_fresh() -> bool:
    p = config.INTL_RESULTS_PATH
    return p.exists() and (time.time() - p.stat().st_mtime) < _MAX_CACHE_AGE_S


def refresh_cache(force: bool = False) -> None:
    if not force and _cache_is_fresh():
        return
    logger.info("downloading international_results from %s", config.INTL_RESULTS_URL)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = requests.get(config.INTL_RESULTS_URL, timeout=60)
    res.raise_for_status()
    config.INTL_RESULTS_PATH.write_bytes(res.content)


def load_results(force_refresh: bool = False) -> pd.DataFrame:
    """Return played international matches with parsed dates and integer scores.

    Columns: date, home_team, away_team, home_score, away_score, neutral.
    """
    refresh_cache(force=force_refresh)
    df = pd.read_csv(config.INTL_RESULTS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = load_results(force_refresh=True)
    logger.info("loaded %d international matches (%s -> %s)",
                len(d), d["date"].min().date(), d["date"].max().date())
