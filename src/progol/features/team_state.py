"""Build a single inference feature row using the SAME logic as training.

The training pipeline builds features over a chronologically sorted dataframe
and uses .shift().ewm(span=5) so each row sees only past matches. For inference
we need an equivalent operation given a (home_id, away_id, date): query history
strictly before that date, then compute the EWMA / Elo / form features.
"""
import pandas as pd
import sqlite3

from src.progol import config
from src.progol.features.elo import update_pair, BASE_RATING
from src.progol.features.rolling import EWMA_SPAN


def _form_to_points(form_str):
    if not form_str or not isinstance(form_str, str):
        return 0.5
    points = 0
    weight = 1.0
    for char in reversed(form_str[-5:]):
        if char == 'W':
            points += 3 * weight
        elif char == 'D':
            points += 1 * weight
        weight *= 0.9
    return points


def compute_elo_table(conn: sqlite3.Connection, before_date: str) -> dict:
    """Walk every match strictly before `before_date` and return team_id -> Elo."""
    q = """
        SELECT date, home_id, away_id, goals_home, goals_away
        FROM matches
        WHERE status = 'FT' AND date < ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(q, conn, params=(before_date,))
    ratings = {}
    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        h = ratings.get(hid, BASE_RATING)
        a = ratings.get(aid, BASE_RATING)
        new_h, new_a = update_pair(h, a, row['goals_home'], row['goals_away'])
        ratings[hid] = new_h
        ratings[aid] = new_a
    return ratings


def _team_history(conn: sqlite3.Connection, team_id: int, before_date: str, limit: int = 30) -> pd.DataFrame:
    q = """
        SELECT date, home_id, away_id, goals_home, goals_away,
               home_shots, away_shots
        FROM matches
        WHERE status = 'FT' AND date < ?
          AND (home_id = ? OR away_id = ?)
        ORDER BY date DESC LIMIT ?
    """
    df = pd.read_sql_query(q, conn, params=(before_date, team_id, team_id, limit))
    if df.empty:
        return df
    df = df.sort_values('date')

    is_home = df['home_id'] == team_id
    df['gf'] = df['goals_home'].where(is_home, df['goals_away'])
    df['ga'] = df['goals_away'].where(is_home, df['goals_home'])
    df['sf'] = df['home_shots'].where(is_home, df['away_shots'])
    df['sa'] = df['away_shots'].where(is_home, df['home_shots'])
    df['opponent_id'] = df['away_id'].where(is_home, df['home_id'])
    df['total_goals'] = df['gf'] + df['ga']
    df['drew'] = (df['gf'] == df['ga']).astype(float)
    return df[['date', 'opponent_id', 'gf', 'ga', 'sf', 'sa', 'total_goals', 'drew']]


def _ewma_last(series: pd.Series, span: int = EWMA_SPAN, min_periods: int = 3) -> float:
    if series.empty or len(series) < min_periods:
        return 0.0
    val = series.ewm(span=span, min_periods=min_periods).mean().iloc[-1]
    return 0.0 if pd.isna(val) else float(val)


def compute_team_ewma_block(conn, team_id, before_date, elo_table) -> dict:
    hist = _team_history(conn, team_id, before_date)
    if hist.empty:
        return {'gf_ewma': 0.0, 'ga_ewma': 0.0, 'sf_ewma': 0.0, 'sa_ewma': 0.0,
                'w_gf_ewma': 0.0, 'w_sf_ewma': 0.0,
                'total_goals_ewma': 0.0, 'drew_ewma': 0.0}

    sos = hist['opponent_id'].map(lambda oid: elo_table.get(int(oid), BASE_RATING) / 1500.0).fillna(1.0)
    hist = hist.assign(w_gf=hist['gf'] * sos, w_sf=hist['sf'] * sos)

    return {
        'gf_ewma': _ewma_last(hist['gf']),
        'ga_ewma': _ewma_last(hist['ga']),
        'sf_ewma': _ewma_last(hist['sf']),
        'sa_ewma': _ewma_last(hist['sa']),
        'w_gf_ewma': _ewma_last(hist['w_gf']),
        'w_sf_ewma': _ewma_last(hist['w_sf']),
        'total_goals_ewma': _ewma_last(hist['total_goals']),
        'drew_ewma': _ewma_last(hist['drew']),
    }


def _team_xg_ewma(conn, team_id, before_date) -> float:
    q = """
        SELECT date, home_id, home_xg, away_xg FROM matches
        WHERE status = 'FT' AND date < ? AND (home_id = ? OR away_id = ?)
          AND home_xg IS NOT NULL
        ORDER BY date DESC LIMIT 30
    """
    df = pd.read_sql_query(q, conn, params=(before_date, team_id, team_id))
    if df.empty:
        return 0.0
    df = df.sort_values('date')
    xg = df['home_xg'].where(df['home_id'] == team_id, df['away_xg'])
    return _ewma_last(xg)


def _h2h_diff(conn, h_id, a_id, before_date) -> int:
    q = """
        SELECT goals_home, goals_away FROM matches
        WHERE date < ? AND status = 'FT'
          AND ((home_id = ? AND away_id = ?) OR (home_id = ? AND away_id = ?))
        ORDER BY date DESC LIMIT 10
    """
    df = pd.read_sql_query(q, conn, params=(before_date, h_id, a_id, a_id, h_id))
    if df.empty:
        return 0
    home_wins = (df['goals_home'] > df['goals_away']).sum()
    away_wins = (df['goals_home'] < df['goals_away']).sum()
    return int(home_wins - away_wins)


def _days_since_last_match(conn, team_id, before_date) -> float:
    q = """
        SELECT date FROM matches
        WHERE status = 'FT' AND date < ? AND (home_id = ? OR away_id = ?)
        ORDER BY date DESC LIMIT 1
    """
    df = pd.read_sql_query(q, conn, params=(before_date, team_id, team_id))
    if df.empty:
        return 7.0
    last = pd.to_datetime(df.iloc[0]['date'])
    cur = pd.to_datetime(before_date)
    return float(max(0.0, (cur - last).total_seconds() / 86400.0))


def _latest_rank_form(conn, team_id, before_date):
    q = """
        SELECT home_rank, home_form, away_rank, away_form, home_id
        FROM matches
        WHERE date < ? AND (home_id = ? OR away_id = ?)
          AND (home_rank IS NOT NULL OR away_rank IS NOT NULL)
        ORDER BY date DESC LIMIT 1
    """
    df = pd.read_sql_query(q, conn, params=(before_date, team_id, team_id))
    if df.empty:
        return 15, "DDDDD"
    row = df.iloc[0]
    if row['home_id'] == team_id:
        return (row['home_rank'] if pd.notna(row['home_rank']) else 15,
                row['home_form'] or "DDDDD")
    return (row['away_rank'] if pd.notna(row['away_rank']) else 15,
            row['away_form'] or "DDDDD")


def build_inference_row(
    conn: sqlite3.Connection,
    home_id: int,
    away_id: int,
    league_id: int,
    date: str,
    venue: str = "Unknown",
    referee: str = "Unknown",
    venue_surface: str = "grass",
    market_probs=None,
    home_injuries: int = 0,
    away_injuries: int = 0,
) -> dict:
    """Returns a dict with keys matching src.progol.config.FEATURE_COLS + categoricals."""
    elo_table = compute_elo_table(conn, date)
    elo_h = elo_table.get(int(home_id), BASE_RATING)
    elo_a = elo_table.get(int(away_id), BASE_RATING)

    h_block = compute_team_ewma_block(conn, home_id, date, elo_table)
    a_block = compute_team_ewma_block(conn, away_id, date, elo_table)

    xg_h = _team_xg_ewma(conn, home_id, date)
    xg_a = _team_xg_ewma(conn, away_id, date)

    h_rank, h_form = _latest_rank_form(conn, home_id, date)
    a_rank, a_form = _latest_rank_form(conn, away_id, date)

    h2h = _h2h_diff(conn, home_id, away_id, date)

    rest_h = _days_since_last_match(conn, home_id, date)
    rest_a = _days_since_last_match(conn, away_id, date)

    if market_probs is None:
        market_probs = (0.45, 0.25, 0.30)

    is_artificial = 1 if venue_surface and 'artificial' in venue_surface.lower() else 0

    return {
        'xg_diff': xg_h - xg_a,
        'elo_diff': elo_h - elo_a,
        'rank_gap': float(a_rank) - float(h_rank),
        'momentum_diff': _form_to_points(h_form) - _form_to_points(a_form),
        'h2h_diff': h2h,
        'is_artificial': is_artificial,
        'gf_ewma_diff': h_block['gf_ewma'] - a_block['gf_ewma'],
        'ga_ewma_diff': h_block['ga_ewma'] - a_block['ga_ewma'],
        'sf_ewma_diff': h_block['sf_ewma'] - a_block['sf_ewma'],
        'sos_gf_diff': h_block['w_gf_ewma'] - a_block['w_gf_ewma'],
        'rest_diff': rest_h - rest_a,
        'total_goals_avg': (h_block['total_goals_ewma'] + a_block['total_goals_ewma']) / 2.0,
        'draw_rate_avg': (h_block['drew_ewma'] + a_block['drew_ewma']) / 2.0,
        'is_cup': 1 if int(league_id) in config.CUP_LEAGUE_IDS else 0,
        'injuries_diff': int(home_injuries) - int(away_injuries),
        'prob_market_h': market_probs[0],
        'prob_market_d': market_probs[1],
        'prob_market_a': market_probs[2],
        'league_id': league_id,
        'venue': venue or "Unknown",
        'referee': referee or "Unknown",
    }
