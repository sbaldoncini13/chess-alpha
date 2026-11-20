import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.sequence_windows import compute_windows_for_game, validate_windows_cover_exactly


import pytest


@pytest.mark.parametrize("n_plies", [1, 5, 10, 78, 125, 126, 127, 128, 129, 200, 255, 256, 257, 258, 400, 500, 561])
@pytest.mark.parametrize("context_length", [128, 256])
def test_full_coverage_no_dup_no_missing(n_plies, context_length):
    r = validate_windows_cover_exactly(n_plies, context_length)
    assert r["coverage_pct"] == 100.0
    assert r["n_duplicates"] == 0
    assert r["n_missing"] == 0
    assert r["n_extra"] == 0
    assert r["eos_covered_exactly_once"]
    assert r["order_ok"]


def test_single_window_short_game():
    windows = compute_windows_for_game(n_plies=10, context_length=256)
    assert len(windows) == 1
    w = windows[0]
    assert w["loss_start"] == 1
    assert w["loss_end"] == 11  # N-1 = (10+2)-1 = 11 (EOS)


def test_multi_window_long_game():
    windows = compute_windows_for_game(n_plies=561, context_length=256)
    assert len(windows) >= 2
    # el ultimo target cubierto por la ultima ventana debe ser el EOS (N-1)
    N = 561 + 2
    assert windows[-1]["loss_end"] == N - 1


def test_no_window_crosses_into_out_of_range_target():
    for n_plies in [1, 128, 256, 561]:
        for C in [128, 256]:
            windows = compute_windows_for_game(n_plies, C)
            N = n_plies + 2
            for w in windows:
                assert w["target_end"] <= N - 1
                assert w["loss_end"] <= N - 1
