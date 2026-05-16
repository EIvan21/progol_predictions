import os
import logging
import numpy as np
import pandas as pd

from src.progol import config
from src.progol import database
from src.progol.features.elo import calculate_elo_ratings
from src.progol.features.rolling import add_rolling_features
from src.progol.features.team_state import _form_to_points

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def get_target(row):
    if row['goals_home'] > row['goals_away']:
        return 0
    elif row['goals_home'] == row['goals_away']:
        return 1
    return 2


def calculate_alpha_features(df):
    logging.info("Building alpha features...")
    pd.set_option('future.no_silent_downcasting', True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df, history_df, _ = calculate_elo_ratings(df)
    df = add_rolling_features(df, history_df)

    h_xg = df[['fixture_id', 'date', 'home_id', 'home_xg']].rename(columns={'home_id': 'team_id', 'home_xg': 'xg'})
    a_xg = df[['fixture_id', 'date', 'away_id', 'away_xg']].rename(columns={'away_id': 'team_id', 'away_xg': 'xg'})
    xg_stats = pd.concat([h_xg, a_xg]).sort_values(['team_id', 'date'])
    # EWMA span=5 matches team_state._team_xg_ewma used at inference time.
    # Previously used .rolling(5).mean() which gave inference vs training
    # drift on the xg_diff feature (drift detector flagged every fixture).
    xg_stats['roll_xg'] = xg_stats.groupby('team_id')['xg'].transform(
        lambda x: x.shift().ewm(span=5, min_periods=3).mean()
    ).fillna(0)

    df = df.merge(
        xg_stats[['fixture_id', 'team_id', 'roll_xg']],
        left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'],
        suffixes=('', '_h'),
    ).drop(columns=['team_id'])
    df = df.merge(
        xg_stats[['fixture_id', 'team_id', 'roll_xg']],
        left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'],
        suffixes=('_h', '_a'),
    ).drop(columns=['team_id'])

    df['xg_diff'] = df['roll_xg_h'] - df['roll_xg_a']
    df['elo_diff'] = df['elo_home'] - df['elo_away']

    df['gf_ewma_diff'] = df['home_gf_ewma'] - df['away_gf_ewma']
    df['ga_ewma_diff'] = df['home_ga_ewma'] - df['away_ga_ewma']
    df['sf_ewma_diff'] = df['home_sf_ewma'] - df['away_sf_ewma']
    df['sos_gf_diff'] = df['home_w_gf_ewma'] - df['away_w_gf_ewma']

    # Draw-prone signals — model used to never produce E as the modal pick
    # because it had no symmetric features. Both are AVERAGES (not diffs)
    # so when both teams are draw-prone or both play low-scoring matches,
    # the value is high and pushes E up regardless of L/V asymmetry.
    df['total_goals_avg'] = (df['home_total_goals_ewma'] + df['away_total_goals_ewma']) / 2.0
    df['draw_rate_avg'] = (df['home_drew_ewma'] + df['away_drew_ewma']) / 2.0

    df['rank_gap'] = df['away_rank'].fillna(15) - df['home_rank'].fillna(15)
    df['momentum_h'] = df['home_form'].apply(_form_to_points)
    df['momentum_a'] = df['away_form'].apply(_form_to_points)
    df['momentum_diff'] = df['momentum_h'] - df['momentum_a']

    df['h2h_diff'] = df['h2h_home_wins'].fillna(0) - df['h2h_away_wins'].fillna(0)

    h_dates = df[['fixture_id', 'date', 'home_id']].rename(columns={'home_id': 'team_id'})
    a_dates = df[['fixture_id', 'date', 'away_id']].rename(columns={'away_id': 'team_id'})
    team_dates = pd.concat([h_dates, a_dates]).sort_values(['team_id', 'date'])
    team_dates['prev_date'] = team_dates.groupby('team_id')['date'].shift(1)
    team_dates['rest_days'] = (team_dates['date'] - team_dates['prev_date']).dt.total_seconds() / 86400.0
    team_dates['rest_days'] = team_dates['rest_days'].fillna(7.0).clip(lower=0.0)
    df = df.merge(team_dates[['fixture_id', 'team_id', 'rest_days']],
                  left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={'rest_days': 'rest_h'}).drop(columns=['team_id'])
    df = df.merge(team_dates[['fixture_id', 'team_id', 'rest_days']],
                  left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={'rest_days': 'rest_a'}).drop(columns=['team_id'])
    df['rest_diff'] = df['rest_h'].fillna(7.0) - df['rest_a'].fillna(7.0)

    df['is_artificial'] = df['venue_surface'].apply(
        lambda x: 1 if x and 'artificial' in x.lower() else 0
    )

    df['is_cup'] = df['league_id'].isin(config.CUP_LEAGUE_IDS).astype(int)

    df['injuries_diff'] = df['home_injuries'].fillna(0) - df['away_injuries'].fillna(0)

    df['target'] = df.apply(get_target, axis=1)

    final_cols = ['fixture_id', 'date', 'target'] + config.CAT_COLS + config.FEATURE_COLS
    processed = df[final_cols].fillna(0)
    processed = processed.replace([np.inf, -np.inf], 0)
    return processed


def process_matches_from_db():
    df = database.get_all_matches_df()
    logging.info(f"Processing {len(df)} matches from database.")
    return calculate_alpha_features(df)


if __name__ == "__main__":
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df = process_matches_from_db()
    df.to_csv(config.TRAIN_CSV, index=False)
    logging.info(f"SUCCESS: wrote {config.TRAIN_CSV}")
