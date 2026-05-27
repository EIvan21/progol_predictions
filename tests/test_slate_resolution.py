"""Tests for Progol slate resolution: NICKNAME_MAP coverage, clean_name,
and resolve_matches threshold behavior.

Guards against the concurso 2333/2334 bug where 6 and 5 fixtures failed
to resolve — some due to missing NICKNAME_MAP entries, others because
the fuzzy threshold (was 75) rejected close-but-valid matches.
"""
import pytest

from src.progol.ingest.get_progol_ids import (
    NICKNAME_MAP,
    clean_name,
    resolve_matches,
)


# --- clean_name + NICKNAME_MAP ------------------------------------------------

# Each tuple: (raw quinielaposible name, expected clean_name output).
# If the assertion fails, a NICKNAME_MAP entry is missing or wrong.
_EXPECTED_MAPPINGS = [
    # MLS — concurso 2333 failures (games 8, 9)
    ("NY R. BULLS", "NEW YORK RED BULLS"),
    ("NY CITY", "NEW YORK CITY"),
    ("HOUSTON", "HOUSTON DYNAMO"),
    ("VANCOUVER", "VANCOUVER WHITECAPS"),
    ("MINNESOTA", "MINNESOTA UNITED"),
    ("SALT LAKE", "REAL SALT LAKE"),
    ("SAN DIEGO", "SAN DIEGO FC"),
    ("CHARLOTTE", "CHARLOTTE FC"),
    ("NEW ENGLAND", "NEW ENGLAND REVOLUTION"),
    ("TORONTO", "TORONTO FC"),
    # Russian Premier — concurso 2333 game 14, 2334 game 12
    ("CSKA", "CSKA MOSCOW"),
    ("LOKOMOTIV", "LOKOMOTIV MOSCOW"),
    ("SPARTAK", "SPARTAK MOSCOW"),
    ("KRASNODAR", "FC KRASNODAR"),
    # Chilean Primera — concurso 2334 game 11
    ("U. CATOLICA", "UNIVERSIDAD CATOLICA"),
    # J-League — concurso 2334 games 13, 14
    ("CEREZO OSAK", "CEREZO OSAKA"),
    ("GAMBA OSAK", "GAMBA OSAKA"),
    ("SHIMIZU", "SHIMIZU S-PULSE"),
    ("OKAYAMA", "FAGIANO OKAYAMA"),
    # Women's — concurso 2334 game 21
    ("BARCELONA F", "BARCELONA FEMENI"),
    ("LYONNES F", "OLYMPIQUE LYONNAIS"),
    # Belgian Jupiler — concurso 2333 game 20
    ("BRUJAS", "BRUGGE"),
    ("UNION SG", "UNION SAINT-GILLOISE"),
    # Scottish Premiership — concurso 2333 game 21
    ("HEARTS", "HEART OF MIDLOTHIAN"),
    # Premier League
    ("MAN. CITY", "MANCHESTER CITY"),
    ("MAN. UNITED", "MANCHESTER UNITED"),
    # Liga MX
    ("C. AZUL", "CRUZ AZUL"),
    ("PUMAS", "UNAM PUMAS"),
    ("CHIVAS", "GUADALAJARA"),
    ("TIGRES", "TIGRES UANL"),
    # National teams (Spanish) — concurso 2335 failures
    ("E.U.A.", "USA"),
    ("NORUEGA", "NORWAY"),
    ("SUECIA", "SWEDEN"),
    ("JAPON", "JAPAN"),
    ("ISLANDIA", "ICELAND"),
    ("CHEQUIA", "CZECH REPUBLIC"),
    ("BRASIL", "BRAZIL"),
    ("PANAMA", "PANAMA"),
    ("A. SAUDITA", "SAUDI ARABIA"),
    ("MEXICO", "MEXICO"),
    ("ECUADOR", "ECUADOR"),
    ("KOSOVO", "KOSOVO"),
    ("ALEMANIA", "GERMANY"),
    ("FRANCIA", "FRANCE"),
    ("INGLATERRA", "ENGLAND"),
    ("HOLANDA", "NETHERLANDS"),
    ("COREA SUR", "SOUTH KOREA"),
    ("MARRUECOS", "MOROCCO"),
    ("AUSTRALIA", "AUSTRALIA"),
    # Brazil Serie B — concurso 2335 games 12, 21
    ("AVAI", "AVAI FC"),
    ("ATL. GO", "ATLETICO GOIANIENSE"),
    # Swedish Allsvenskan — concurso 2335 game 14
    ("DEGERFORS", "DEGERFORS IF"),
    ("BROMMAPOJ", "BROMMAPOJKARNA"),
]


@pytest.mark.parametrize("raw,expected", _EXPECTED_MAPPINGS,
                         ids=[f"{r}->{e}" for r, e in _EXPECTED_MAPPINGS])
def test_clean_name_maps_progol_shortform(raw, expected):
    assert clean_name(raw) == expected


# --- resolve_matches threshold ------------------------------------------------

def _fake_fixture(home, away, fixture_id):
    return {
        'teams': {
            'home': {'name': home},
            'away': {'name': away},
        },
        'fixture': {'id': fixture_id},
    }


def test_resolve_threshold_70_catches_previously_missed():
    """At threshold=70, matches that scored 70-74 (like CELTIC vs HEARTS=72)
    now resolve instead of FAILED."""
    api_data = [
        _fake_fixture("Heart of Midlothian", "Celtic", 99001),
    ]
    games = resolve_matches(
        [("CELTIC", "HEARTS")],
        api_data,
        threshold=70,
    )
    assert games[0]['fixture_id'] is not None


def test_resolve_below_threshold_still_fails():
    """Matches below 70 must still fail — low threshold shouldn't accept garbage."""
    api_data = [
        _fake_fixture("Totally Different Team", "Another Random Club", 88001),
    ]
    games = resolve_matches(
        [("CELTIC", "HEARTS")],
        api_data,
        threshold=70,
    )
    assert games[0]['fixture_id'] is None


def test_resolve_preserves_game_number():
    """game_number must be 1-indexed position in the scraped list."""
    api_data = [
        _fake_fixture("Cerezo Osaka", "Gamba Osaka", 55001),
        _fake_fixture("Shimizu S-Pulse", "Fagiano Okayama", 55002),
    ]
    games = resolve_matches(
        [("CEREZO OSAK", "GAMBA OSAK"), ("SHIMIZU", "OKAYAMA")],
        api_data,
        threshold=70,
    )
    assert games[0]['game_number'] == 1
    assert games[0]['fixture_id'] == 55001
    assert games[1]['game_number'] == 2
    assert games[1]['fixture_id'] == 55002


def test_resolve_finds_match_regardless_of_home_away_order():
    """token_sort_ratio makes matching order-independent, so even when
    Progol lists home/away reversed vs the API the fixture is found."""
    api_data = [
        _fake_fixture("Brugge", "Union Saint-Gilloise", 77001),
    ]
    games = resolve_matches(
        [("UNION SG", "BRUJAS")],
        api_data,
        threshold=70,
    )
    assert games[0]['fixture_id'] == 77001
