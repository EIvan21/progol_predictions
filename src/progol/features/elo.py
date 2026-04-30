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


def calculate_elo_ratings(df, k_factor=K_FACTOR, base_rating=BASE_RATING):
    df = df.sort_values('date').copy()
    team_ratings = {}
    rating_history = []
    elo_h_col, elo_a_col = [], []

    for _, row in df.iterrows():
        hid, aid = row['home_id'], row['away_id']
        date = row['date']
        h_rating = team_ratings.get(hid, base_rating)
        a_rating = team_ratings.get(aid, base_rating)

        elo_h_col.append(h_rating)
        elo_a_col.append(a_rating)

        rating_history.append({'date': date, 'team_id': hid, 'rating': h_rating})
        rating_history.append({'date': date, 'team_id': aid, 'rating': a_rating})

        new_h, new_a = update_pair(h_rating, a_rating, row['goals_home'], row['goals_away'], k=k_factor)
        team_ratings[hid] = new_h
        team_ratings[aid] = new_a

    df['elo_home'] = elo_h_col
    df['elo_away'] = elo_a_col

    history_df = pd.DataFrame(rating_history).drop_duplicates(subset=['date', 'team_id'], keep='last')
    return df, history_df, team_ratings
