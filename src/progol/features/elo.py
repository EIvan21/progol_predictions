import pandas as pd

K_FACTOR = 20
BASE_RATING = 1500
HOME_ADV = 75


def expected_home(h_rating, a_rating, home_adv=HOME_ADV):
    dr = a_rating - (h_rating + home_adv)
    return 1 / (1 + 10 ** (dr / 400))


def update_pair(h_rating, a_rating, goals_home, goals_away, k=K_FACTOR):
    e_h = expected_home(h_rating, a_rating)
    if goals_home > goals_away:
        result = 1.0
    elif goals_home == goals_away:
        result = 0.5
    else:
        result = 0.0
    new_h = h_rating + k * (result - e_h)
    new_a = a_rating + k * ((1 - result) - (1 - e_h))
    return new_h, new_a


def _league_mean(team_ratings: dict, team_to_league: dict, league_id, base_rating: float) -> float:
    """Mean Elo of all teams whose most-recent league matches `league_id`.
    Falls back to base_rating if no team in that league has been seen yet.
    Used for cold-start of new teams: a Liga MX promotee starting at the
    Liga MX mean (~1450) is a far better prior than the global 1500."""
    if league_id is None:
        return base_rating
    ratings = [r for tid, r in team_ratings.items() if team_to_league.get(tid) == league_id]
    if not ratings:
        return base_rating
    return sum(ratings) / len(ratings)


def calculate_elo_ratings(df, k_factor=K_FACTOR, base_rating=BASE_RATING):
    df = df.sort_values('date').copy()
    team_ratings = {}
    # Track each team's most recent league so cold-start initialization
    # for new teams uses the right per-league prior. Promotions / relegations
    # are rare enough that "most recent" is a good approximation; the rating
    # itself carries forward across league moves so promoted teams keep their
    # earned Elo rather than resetting.
    team_to_league = {}
    rating_history = []
    elo_h_col, elo_a_col = [], []

    has_league_id = 'league_id' in df.columns

    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        lid = row['league_id'] if has_league_id else None
        date = row['date']

        if hid in team_ratings:
            h_rating = team_ratings[hid]
        else:
            h_rating = _league_mean(team_ratings, team_to_league, lid, base_rating)
        if aid in team_ratings:
            a_rating = team_ratings[aid]
        else:
            a_rating = _league_mean(team_ratings, team_to_league, lid, base_rating)

        elo_h_col.append(h_rating)
        elo_a_col.append(a_rating)

        rating_history.append({'date': date, 'team_id': hid, 'rating': h_rating})
        rating_history.append({'date': date, 'team_id': aid, 'rating': a_rating})

        new_h, new_a = update_pair(h_rating, a_rating, row['goals_home'], row['goals_away'], k=k_factor)
        team_ratings[hid] = new_h
        team_ratings[aid] = new_a
        if lid is not None:
            team_to_league[hid] = lid
            team_to_league[aid] = lid

    df['elo_home'] = elo_h_col
    df['elo_away'] = elo_a_col

    history_df = pd.DataFrame(rating_history).drop_duplicates(subset=['date', 'team_id'], keep='last')
    return df, history_df, team_ratings
