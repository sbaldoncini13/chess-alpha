"""Correccion 3: tests dedicados a engine_criticality.py (senales MultiPV
puras, sin depender de un Stockfish real -- construimos topk/policy_rows
sinteticos con la MISMA forma que engine_records.py produce)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from engine_criticality import engine_criticality_row, engine_criticality_score, engine_tactical_or_critical  # noqa: E402

CFG = {"gap_saturation_cp": 200, "w_gap": 0.6, "w_policy_concentration": 0.4, "threshold": 0.5}


def _topk(*evals):
    """evals: lista de (eval_type, unified_cp) en orden de rank creciente."""
    rows = []
    for i, (etype, unified) in enumerate(evals, start=1):
        rows.append({
            "row_id": 1, "rank": i, "move_id": 100 + i, "move_uci": f"m{i}",
            "eval_type": etype, "eval_cp": float(unified) if etype == "cp" else None,
            "mate_in": None if etype == "cp" else unified, "eval_unified_cp_equivalent": unified, "depth": 10,
        })
    return rows


def _policy(*weights):
    return [{"row_id": 1, "move_id": 100 + i, "probability": w, "rank": i}
            for i, w in enumerate(weights, start=1)]


def test_small_gap_top1_top2_low_score():
    topk = _topk(("cp", 40), ("cp", 35), ("cp", 20))
    policy = _policy(0.4, 0.35, 0.25)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert any_mate is False
    assert gap_cp == 5.0
    gap_term = 5.0 / 200
    expected = 0.6 * gap_term + 0.4 * 0.4
    assert score == pytest.approx(expected)
    assert score < 0.5


def test_large_gap_top1_top2_saturates_gap_term():
    topk = _topk(("cp", 500), ("cp", 100), ("cp", 50))
    policy = _policy(0.9, 0.08, 0.02)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert gap_cp == 400.0
    # gap_term clampeado a 1.0 (400/200 > 1)
    expected = 0.6 * 1.0 + 0.4 * 0.9
    assert score == pytest.approx(expected)
    assert score >= CFG["threshold"]


def test_unique_clearly_good_move_high_policy_concentration():
    # gap grande Y policy muy concentrada en rank=1 -> score alto
    topk = _topk(("cp", 300), ("cp", -50))
    policy = _policy(0.99, 0.01)
    score, _, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert policy_top1 == pytest.approx(0.99)
    assert score > 0.9


def test_mate_positive_forces_maximal_criticality():
    topk = _topk(("mate", 99998), ("cp", 40))  # mate en 2 a favor del que mueve
    policy = _policy(0.95, 0.05)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert score == 1.0
    assert any_mate is True
    assert gap_cp is None  # no se calcula gap cuando hay mate


def test_mate_negative_also_forces_maximal_criticality():
    # rank=1 mate en contra del que mueve (mate_in negativo, unified negativo)
    topk = _topk(("mate", -99997), ("cp", -300))
    policy = _policy(0.6, 0.4)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert score == 1.0
    assert any_mate is True


def test_multipv_incomplete_only_two_of_three_requested():
    # engine.analyse devolvio menos candidatos que multipv configurado
    topk = _topk(("cp", 60), ("cp", 55))
    policy = _policy(0.55, 0.45)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert any_mate is False
    assert gap_cp == 5.0
    assert score is not None  # no debe fallar por MultiPV incompleto


def test_single_candidate_gap_term_zero_policy_term_full():
    topk = _topk(("cp", 40))
    policy = _policy(1.0)
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy, CFG)
    assert gap_cp is None
    assert policy_top1 == 1.0
    assert score == pytest.approx(CFG["w_policy_concentration"] * 1.0)


def test_missing_values_empty_topk_returns_none():
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score([], [], CFG)
    assert score is None
    assert any_mate is False
    assert gap_cp is None
    assert policy_top1 is None
    assert engine_tactical_or_critical(score, CFG) is None


def test_engine_criticality_is_deterministic():
    topk = _topk(("cp", 80), ("cp", 10))
    policy = _policy(0.8, 0.2)
    a = engine_criticality_score(topk, policy, CFG)
    b = engine_criticality_score(topk, policy, CFG)
    assert a == b


def test_engine_criticality_row_shape():
    topk = _topk(("cp", 80), ("cp", 10))
    policy = _policy(0.8, 0.2)
    row = engine_criticality_row(42, topk, policy, CFG)
    assert row["row_id"] == 42
    assert 0.0 <= row["engine_criticality_score"] <= 1.0
    assert isinstance(row["engine_tactical_or_critical"], bool)


def test_engine_criticality_never_touches_base_criticality_module():
    """No debe existir ningun acoplamiento con difficulty_criticality.py --
    son conceptos y columnas separados por diseno."""
    import engine_criticality
    assert "difficulty_criticality" not in dir(engine_criticality)
    with open(os.path.join(os.path.dirname(__file__), "..", "scripts", "engine_criticality.py"), encoding="utf-8") as f:
        src = f.read()
    assert "import difficulty_criticality" not in src
    assert "from difficulty_criticality" not in src
