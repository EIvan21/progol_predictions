"""Regression tests for the Progol slate alignment helpers in predict.py.

The bug we're guarding against: when some Progol fixtures fail to resolve
(API miss, score not posted yet, etc.), the resolved subset is shorter
than the slate. Old code used `enumerate(match_ids, start=1)` which
assigned each resolved match to a contiguous 1..N range — that shifted
the concurso writes + plan-optimizer indices by the size of the gap,
causing rows like "NY Red Bulls" to show Nashville's L/E/V probabilities.

The fix is `align_slots(match_ids, game_numbers)`: when get_progol_ids
passes the parallel `game_numbers` array, we zip instead of enumerate
so the ORIGINAL slot 1..21 is preserved per match_id.
"""
import numpy as np

from src.progol.modeling.predict import (
    DEFAULT_SLOT_PRIOR,
    align_slots,
    probs_to_slot_array,
)


# --- align_slots -----------------------------------------------------------

def test_align_slots_preserves_original_game_numbers():
    """When game_numbers + match_ids both present, pair them directly —
    a missing slot in the middle doesn't shift later positions."""
    match_ids = [1001, 1003, 1004, 1007]
    game_numbers = [1, 3, 4, 7]
    pairs = align_slots(match_ids, game_numbers)
    assert pairs == [(1, 1001), (3, 1003), (4, 1004), (7, 1007)]


def test_align_slots_revancha_slots_above_14():
    """Slots can be 1..21 (14 main + 7 revancha); helper must not clamp."""
    match_ids = [200, 201, 202]
    game_numbers = [13, 14, 21]
    pairs = align_slots(match_ids, game_numbers)
    assert pairs == [(13, 200), (14, 201), (21, 202)]


def test_align_slots_falls_back_to_enumerate_when_missing():
    """Legacy slate JSON had no game_numbers field — must not crash, just
    assign contiguous 1..N. Documented behavior change is the user's
    explicit choice (they should re-scrape to get the alignment fix)."""
    pairs = align_slots([10, 20, 30], None)
    assert pairs == [(1, 10), (2, 20), (3, 30)]


def test_align_slots_falls_back_when_lengths_mismatch():
    """Defensive: if get_progol_ids ever writes a mismatched parallel
    array (e.g., partial scrape lost rows), don't silently zip a wrong
    pairing — fall back to enumerate and let downstream see contiguous
    slots. Loud failure modes (concurso row mismatch) would surface
    faster than a silent off-by-one."""
    pairs = align_slots([10, 20, 30], [5, 9])  # len mismatch
    assert pairs == [(1, 10), (2, 20), (3, 30)]


def test_align_slots_empty_inputs():
    """Empty slate is legal (scrape returned zero fixtures); helper must
    not crash."""
    assert align_slots([], []) == []
    assert align_slots([], None) == []


# --- probs_to_slot_array ---------------------------------------------------

def test_probs_to_slot_array_full_slate():
    """Every slot has a result → array values match input verbatim."""
    results = [
        {'gn': i, 'h': 0.5, 'd': 0.2, 'v': 0.3} for i in range(1, 15)
    ]
    arr = probs_to_slot_array(results, n_slots=14)
    assert arr.shape == (14, 3)
    np.testing.assert_array_almost_equal(arr[0], [0.5, 0.2, 0.3])
    np.testing.assert_array_almost_equal(arr[13], [0.5, 0.2, 0.3])


def test_probs_to_slot_array_missing_slots_use_default_prior():
    """The regression scenario: only slots 1, 3, 4, 7 resolved.
    Slots 2, 5, 6, 8..14 must be the default prior — and slot 3 must
    NOT shift to position 2, etc."""
    results = [
        {'gn': 1, 'h': 0.6, 'd': 0.2, 'v': 0.2},
        {'gn': 3, 'h': 0.1, 'd': 0.3, 'v': 0.6},
        {'gn': 4, 'h': 0.4, 'd': 0.3, 'v': 0.3},
        {'gn': 7, 'h': 0.9, 'd': 0.05, 'v': 0.05},
    ]
    arr = probs_to_slot_array(results, n_slots=14)
    assert arr.shape == (14, 3)
    # Resolved slots land in the right rows (index = slot - 1).
    np.testing.assert_array_almost_equal(arr[0], [0.6, 0.2, 0.2])
    np.testing.assert_array_almost_equal(arr[2], [0.1, 0.3, 0.6])
    np.testing.assert_array_almost_equal(arr[3], [0.4, 0.3, 0.3])
    np.testing.assert_array_almost_equal(arr[6], [0.9, 0.05, 0.05])
    # Missing slots use the default prior.
    np.testing.assert_array_almost_equal(arr[1], list(DEFAULT_SLOT_PRIOR))
    np.testing.assert_array_almost_equal(arr[4], list(DEFAULT_SLOT_PRIOR))
    np.testing.assert_array_almost_equal(arr[13], list(DEFAULT_SLOT_PRIOR))


def test_probs_to_slot_array_ignores_revancha_when_n_slots_14():
    """Slate has main + revancha; budget optimizer wants only the 14
    main slots. Revancha results (gn > 14) must NOT bleed into the
    returned array."""
    results = [
        {'gn': 1, 'h': 0.7, 'd': 0.2, 'v': 0.1},
        {'gn': 15, 'h': 0.1, 'd': 0.1, 'v': 0.8},  # revancha, must be dropped
        {'gn': 21, 'h': 0.2, 'd': 0.6, 'v': 0.2},  # revancha, must be dropped
    ]
    arr = probs_to_slot_array(results, n_slots=14)
    assert arr.shape == (14, 3)
    np.testing.assert_array_almost_equal(arr[0], [0.7, 0.2, 0.1])
    # All other slots are the default prior — nothing leaked from gn=15 or gn=21.
    for i in range(1, 14):
        np.testing.assert_array_almost_equal(arr[i], list(DEFAULT_SLOT_PRIOR))


def test_probs_to_slot_array_revancha_when_n_slots_21():
    """Caller can request the full slate including revancha by passing
    n_slots=21 — helper accommodates without changing semantics."""
    results = [
        {'gn': 1, 'h': 0.5, 'd': 0.3, 'v': 0.2},
        {'gn': 17, 'h': 0.2, 'd': 0.2, 'v': 0.6},
    ]
    arr = probs_to_slot_array(results, n_slots=21)
    assert arr.shape == (21, 3)
    np.testing.assert_array_almost_equal(arr[0], [0.5, 0.3, 0.2])
    np.testing.assert_array_almost_equal(arr[16], [0.2, 0.2, 0.6])
    np.testing.assert_array_almost_equal(arr[20], list(DEFAULT_SLOT_PRIOR))


def test_probs_to_slot_array_custom_default_prior():
    """Default prior should be tunable for backtests / ablation."""
    custom = (0.34, 0.33, 0.33)  # uniform
    arr = probs_to_slot_array([], n_slots=14, default_prior=custom)
    for i in range(14):
        np.testing.assert_array_almost_equal(arr[i], list(custom))


def test_probs_to_slot_array_empty_results():
    """No results → all slots default. Used to be a silent ValueError on
    np.array([]) — verify it returns a sane (14, 3) array."""
    arr = probs_to_slot_array([], n_slots=14)
    assert arr.shape == (14, 3)
    for i in range(14):
        np.testing.assert_array_almost_equal(arr[i], list(DEFAULT_SLOT_PRIOR))
