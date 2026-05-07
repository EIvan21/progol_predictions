import pandas as pd

EWMA_SPAN = 5


def add_rolling_features(df, history_df=None, span=EWMA_SPAN):
    df = df.sort_values('date')

    h = df[['fixture_id', 'date', 'home_id', 'away_id', 'goals_home', 'goals_away', 'home_shots', 'away_shots']].copy()
    h.columns = ['fixture_id', 'date', 'team_id', 'opponent_id', 'gf', 'ga', 'sf', 'sa']

    a = df[['fixture_id', 'date', 'away_id', 'home_id', 'goals_away', 'goals_home', 'away_shots', 'home_shots']].copy()
    a.columns = ['fixture_id', 'date', 'team_id', 'opponent_id', 'gf', 'ga', 'sf', 'sa']

    stats = pd.concat([h, a]).sort_values(['team_id', 'date'])

    # Empate-prone signals: total goals per match (low scoring -> draws likely)
    # and per-match draw indicator (1 if gf == ga). EWMA'd downstream so the
    # model sees recent tendencies, not the team's lifetime average.
    stats['total_goals'] = stats['gf'] + stats['ga']
    stats['drew'] = (stats['gf'] == stats['ga']).astype(float)

    if history_df is not None:
        stats = stats.merge(
            history_df, left_on=['date', 'opponent_id'], right_on=['date', 'team_id'],
            suffixes=('', '_opp'), how='left',
        )
        stats['opp_rating'] = stats['rating'].fillna(1500)
    else:
        stats['opp_rating'] = 1500

    stats['sos_factor'] = stats['opp_rating'] / 1500.0
    stats['w_gf'] = stats['gf'] * stats['sos_factor']
    stats['w_sf'] = stats['sf'] * stats['sos_factor']

    cols_to_avg = ['gf', 'ga', 'sf', 'sa', 'w_gf', 'w_sf', 'total_goals', 'drew']
    grouped = stats.groupby('team_id')[cols_to_avg]

    ewma = grouped.apply(lambda x: x.shift().ewm(span=span, min_periods=3).mean())
    if isinstance(ewma.index, pd.MultiIndex):
        ewma = ewma.reset_index(level=0, drop=True)
    stats = stats.join(ewma, rsuffix='_ewma')

    final_cols = ['fixture_id', 'team_id', 'gf_ewma', 'ga_ewma', 'sf_ewma', 'sa_ewma',
                  'w_gf_ewma', 'w_sf_ewma', 'total_goals_ewma', 'drew_ewma']

    df = df.merge(stats[final_cols], left_on=['fixture_id', 'home_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={c: f'home_{c}' for c in final_cols if c not in ['fixture_id', 'team_id']})
    df = df.drop(columns=['team_id'])

    df = df.merge(stats[final_cols], left_on=['fixture_id', 'away_id'], right_on=['fixture_id', 'team_id'], how='left')
    df = df.rename(columns={c: f'away_{c}' for c in final_cols if c not in ['fixture_id', 'team_id']})
    df = df.drop(columns=['team_id'])

    return df
