"""Correcciones 1+2: tests de engine_records.py (forma de las filas
persistidas) + invariantes de Top-K/Soft Policy contra un Stockfish real
cuando esta disponible."""
import os
import sys

import chess
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from engine_records import eval_row_from_result, policy_rows_from_topk, topk_rows_from_analysis  # noqa: E402
from stockfish_enrichment import CP_MATE_SCORE, StockfishEnricher, softmax_policy_from_topk  # noqa: E402

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")
requires_stockfish = pytest.mark.skipif(not STOCKFISH_PATH, reason="STOCKFISH_PATH no configurado")
QUALITY_THRESHOLDS = {"best": 5, "excellent": 20, "good": 60, "inaccuracy": 120, "mistake": 300}


def _synthetic_analysis():
    return {
        "fen": chess.STARTING_FEN, "depth": 10, "multipv": 3,
        "topk": [
            {"rank": 1, "move": chess.Move.from_uci("e2e4"), "move_id": 1, "move_uci": "e2e4",
             "score_cp": 45, "is_mate": False, "mate_in": None},
            {"rank": 2, "move": chess.Move.from_uci("g1f3"), "move_id": 2, "move_uci": "g1f3",
             "score_cp": 30, "is_mate": False, "mate_in": None},
            {"rank": 3, "move": chess.Move.from_uci("d2d4"), "move_id": 3, "move_uci": "d2d4",
             "score_cp": 20, "is_mate": False, "mate_in": None},
        ],
        "best_move": chess.Move.from_uci("e2e4"), "best_move_id": 1, "best_move_uci": "e2e4", "best_eval_cp": 45,
    }


def _synthetic_result(human_eval_after_cp, centipawn_loss=15, move_quality="excellent"):
    a = _synthetic_analysis()
    a.update({
        "human_move_uci": "e2e4", "human_eval_after_cp": human_eval_after_cp,
        "centipawn_loss": centipawn_loss, "move_quality": move_quality,
    })
    return a


def test_topk_rows_from_analysis_shape_and_cp_decomposition():
    rows = topk_rows_from_analysis(row_id=7, analysis=_synthetic_analysis(), depth=10)
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert all(r["row_id"] == 7 for r in rows)
    assert rows[0]["eval_type"] == "cp"
    assert rows[0]["eval_cp"] == 45.0
    assert rows[0]["mate_in"] is None
    assert rows[0]["eval_unified_cp_equivalent"] == 45


def test_topk_rows_from_analysis_mate_decomposition():
    analysis = _synthetic_analysis()
    analysis["topk"][0]["is_mate"] = True
    analysis["topk"][0]["mate_in"] = 2
    analysis["topk"][0]["score_cp"] = CP_MATE_SCORE - 2
    rows = topk_rows_from_analysis(row_id=1, analysis=analysis, depth=10)
    assert rows[0]["eval_type"] == "mate"
    assert rows[0]["eval_cp"] is None
    assert rows[0]["mate_in"] == 2
    assert rows[0]["eval_unified_cp_equivalent"] == CP_MATE_SCORE - 2


def test_policy_rows_from_topk_sums_to_one_and_carries_rank():
    analysis = _synthetic_analysis()
    rows = policy_rows_from_topk(row_id=7, topk=analysis["topk"], temperature_cp=150.0)
    assert len(rows) == 3
    total = sum(r["probability"] for r in rows)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(r["row_id"] == 7 for r in rows)
    move_id_to_rank = {r["move_id"]: r["rank"] for r in rows}
    assert move_id_to_rank[1] == 1  # move_id=1 (mejor candidato) es rank=1


def test_policy_rows_empty_topk_empty_policy():
    assert policy_rows_from_topk(row_id=1, topk=[], temperature_cp=150.0) == []


def test_eval_row_from_result_cp_case():
    result = _synthetic_result(human_eval_after_cp=30)
    row = eval_row_from_result(row_id=7, result=result, depth=10, multipv_configured=3)
    assert row["best_eval_type"] == "cp"
    assert row["best_eval_cp"] == 45.0
    assert row["best_eval_mate_in"] is None
    assert row["human_eval_type"] == "cp"
    assert row["human_eval_cp"] == 30.0
    assert row["human_eval_mate_in"] is None
    assert row["centipawn_loss"] == 15
    assert row["multipv_returned"] == 3


def test_eval_row_from_result_human_mate_case():
    # el humano deja un mate en 3 a su favor tras su jugada (perspectiva ya invertida)
    mate_in = 3
    unified = CP_MATE_SCORE - mate_in
    result = _synthetic_result(human_eval_after_cp=unified)
    row = eval_row_from_result(row_id=1, result=result, depth=10, multipv_configured=3)
    assert row["human_eval_type"] == "mate"
    assert row["human_eval_cp"] is None
    assert row["human_eval_mate_in"] == mate_in


def test_eval_row_from_result_human_mate_against_case():
    mate_in = -2
    unified = -(CP_MATE_SCORE - abs(mate_in))
    result = _synthetic_result(human_eval_after_cp=unified)
    row = eval_row_from_result(row_id=1, result=result, depth=10, multipv_configured=3)
    assert row["human_eval_type"] == "mate"
    assert row["human_eval_mate_in"] == mate_in


def test_eval_row_from_result_no_topk_best_fields_none():
    result = _synthetic_result(human_eval_after_cp=10)
    result["topk"] = []
    result["best_move_id"] = None
    result["best_move_uci"] = None
    row = eval_row_from_result(row_id=1, result=result, depth=10, multipv_configured=3)
    assert row["best_eval_type"] is None
    assert row["multipv_returned"] == 0


# ---------- invariantes contra Stockfish real ----------

@requires_stockfish
def test_topk_invariants_real_engine():
    with StockfishEnricher(STOCKFISH_PATH, depth=8, multipv=3) as sf:
        fen = chess.STARTING_FEN
        result = sf.evaluate_human_move(fen, "e2e4", QUALITY_THRESHOLDS)
        rows = topk_rows_from_analysis(row_id=0, analysis=result, depth=8)

        ranks = [r["rank"] for r in rows]
        assert ranks[0] == 1
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)  # unicos
        assert len(rows) <= 3  # <= multipv configurado

        assert rows[0]["move_uci"] == result["best_move_uci"]

        board = chess.Board(fen)
        legal_uci = {m.uci() for m in board.legal_moves}
        for r in rows:
            assert r["move_uci"] in legal_uci


@requires_stockfish
def test_policy_and_topk_persistence_round_trip_real_engine():
    with StockfishEnricher(STOCKFISH_PATH, depth=8, multipv=3) as sf:
        fen = chess.STARTING_FEN
        result = sf.evaluate_human_move(fen, "e2e4", QUALITY_THRESHOLDS)
        topk_rows = topk_rows_from_analysis(row_id=5, analysis=result, depth=8)
        policy_rows = policy_rows_from_topk(row_id=5, topk=result["topk"], temperature_cp=150.0)

        assert len(policy_rows) == len(topk_rows)
        total = sum(p["probability"] for p in policy_rows)
        assert total == pytest.approx(1.0, abs=1e-5)
        assert all(p["probability"] >= 0.0 for p in policy_rows)
        assert not any(p["probability"] != p["probability"] for p in policy_rows)  # no NaN

        topk_move_ids = {r["move_id"] for r in topk_rows}
        assert all(p["move_id"] in topk_move_ids for p in policy_rows)
