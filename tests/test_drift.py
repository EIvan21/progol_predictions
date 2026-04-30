import pandas as pd

from src.progol.utils import drift


def test_fit_and_check_row_no_flags_when_in_distribution():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5]})
    stats = drift.fit_stats(df, ['a'])
    assert drift.check_row({'a': 3.0}, stats) == {}


def test_check_row_flags_outliers():
    df = pd.DataFrame({'a': [1.0, 1.1, 0.9, 1.05, 0.95]})
    stats = drift.fit_stats(df, ['a'])
    flags = drift.check_row({'a': 50.0}, stats, z_threshold=4.0)
    assert 'a' in flags
