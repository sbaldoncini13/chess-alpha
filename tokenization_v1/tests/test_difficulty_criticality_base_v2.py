"""Correccion 4: tests dedicados a difficulty_score/criticality_score BASE
(sin motor). Validan el CONTRATO ACTUAL de difficulty_criticality.py -- no
se cambia la formula para satisfacer estos tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from difficulty_criticality import (  # noqa: E402
    criticality_score, difficulty_bucket, difficulty_score, is_tactical_or_critical,
)

W = {"w_inverse_legal_moves": 0.35, "w_is_check": 0.15, "w_forced_move": 0.30, "w_phase_endgame_bonus": 0.20}
CRIT_T = {"forced_move_max_legal_moves": 1, "low_branching_max_legal_moves": 3, "big_eval_swing_cp": 150}


# ---------- DIFFICULTY ----------

def test_difficulty_score_range_always_in_0_1():
    for lmc in [0, 1, 3, 20, 40, 60, 200]:
        for is_check in (True, False):
            for phase in ("opening", "middlegame", "endgame"):
                score = difficulty_score(lmc, is_check, phase, W)
                assert 0.0 <= score <= 1.0


def test_difficulty_simple_position_low_score():
    # muchas jugadas legales, sin jaque, apertura -> dificultad baja
    score = difficulty_score(30, False, "opening", W)
    expected = W["w_inverse_legal_moves"] * (1 - 30 / 60)
    assert score == pytest.approx(expected)
    assert score < 0.33


def test_difficulty_many_legal_moves_lower_than_few_legal_moves():
    many = difficulty_score(40, False, "opening", W)
    few = difficulty_score(2, False, "opening", W)
    assert many < few


def test_difficulty_check_adds_fixed_delta():
    base = difficulty_score(20, False, "opening", W)
    with_check = difficulty_score(20, True, "opening", W)
    assert with_check - base == pytest.approx(W["w_is_check"])


def test_difficulty_forced_move_adds_fixed_delta_and_saturates_inverse_mobility():
    # legal_move_count == 1 activa TANTO el termino de movilidad inversa
    # (practicamente maxima, 1 - 1/60) COMO el termino w_forced_move
    forced = difficulty_score(1, False, "opening", W)
    expected = W["w_inverse_legal_moves"] * (1 - 1 / 60) + W["w_forced_move"]
    assert forced == pytest.approx(expected)


def test_difficulty_endgame_bonus_adds_fixed_delta():
    mid = difficulty_score(20, False, "middlegame", W)
    end = difficulty_score(20, False, "endgame", W)
    assert end - mid == pytest.approx(W["w_phase_endgame_bonus"])


def test_difficulty_is_deterministic():
    a = difficulty_score(15, True, "endgame", W)
    b = difficulty_score(15, True, "endgame", W)
    assert a == b


def test_difficulty_bucket_boundaries():
    assert difficulty_bucket(0.0) == "low"
    assert difficulty_bucket(0.32) == "low"
    assert difficulty_bucket(0.33) == "medium"
    assert difficulty_bucket(0.65) == "medium"
    assert difficulty_bucket(0.66) == "high"
    assert difficulty_bucket(1.0) == "high"


def test_difficulty_bucket_matches_score_for_forced_check_endgame():
    score = difficulty_score(1, True, "endgame", W)
    # 0.35*(59/60) + 0.15 + 0.30 + 0.20 = 0.9941666...
    assert score == pytest.approx(0.9941666666666666)
    assert difficulty_bucket(score) == "high"


# ---------- CRITICALITY BASE ----------

def test_criticality_score_range_always_in_0_1():
    for lmc in [0, 1, 2, 3, 4, 10, 40]:
        for is_check in (True, False):
            score = criticality_score(lmc, is_check, CRIT_T)
            assert 0.0 <= score <= 1.0


def test_criticality_normal_position_low_score():
    # muchas jugadas legales, sin jaque -> nada critico
    score = criticality_score(30, False, CRIT_T)
    assert score == 0.0
    assert is_tactical_or_critical(score) is False


def test_criticality_marked_critical_position():
    # pocas jugadas legales + jaque + forzado -> critico
    score = criticality_score(1, True, CRIT_T)
    assert score == pytest.approx(0.5 + 0.3 + 0.2)
    assert is_tactical_or_critical(score) is True


def test_criticality_low_branching_alone_reaches_threshold():
    # legal_move_count=2 <= low_branching_max(3), sin jaque, no forzado (2>1)
    score = criticality_score(2, False, CRIT_T)
    assert score == pytest.approx(0.5)
    assert is_tactical_or_critical(score) is True  # >= 0.5 exactamente


def test_criticality_check_alone_does_not_reach_threshold():
    # jaque solo (muchas jugadas legales) no alcanza el umbral 0.5
    score = criticality_score(30, True, CRIT_T)
    assert score == pytest.approx(0.3)
    assert is_tactical_or_critical(score) is False


def test_criticality_more_legal_moves_never_increases_score():
    tight = criticality_score(1, False, CRIT_T)
    loose = criticality_score(30, False, CRIT_T)
    assert loose <= tight


def test_criticality_check_increases_score_relative_to_no_check():
    without = criticality_score(10, False, CRIT_T)
    with_check = criticality_score(10, True, CRIT_T)
    assert with_check > without
    assert with_check - without == pytest.approx(0.3)


def test_criticality_is_deterministic():
    a = criticality_score(3, True, CRIT_T)
    b = criticality_score(3, True, CRIT_T)
    assert a == b


def test_tactical_or_critical_threshold_boundary_is_inclusive():
    assert is_tactical_or_critical(0.5) is True
    assert is_tactical_or_critical(0.4999999) is False
