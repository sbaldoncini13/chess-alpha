"""PASO 21-27 (best move, MultiPV, score perspective, mate score, cp loss).
Se saltan automaticamente si STOCKFISH_PATH no esta configurado en el
entorno -- el pipeline base NUNCA depende de que estas pruebas corran."""
import os
import sys

import chess
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from stockfish_enrichment import (  # noqa: E402
    StockfishEnricher, classify_move_quality, softmax_policy_from_topk,
)

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")
requires_stockfish = pytest.mark.skipif(not STOCKFISH_PATH, reason="STOCKFISH_PATH no configurado en el entorno")

QUALITY_THRESHOLDS = {"best": 5, "excellent": 20, "good": 60, "inaccuracy": 120, "mistake": 300}


BACK_RANK_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


@requires_stockfish
def test_best_move_matte_in_one_found():
    with StockfishEnricher(STOCKFISH_PATH, depth=8, multipv=1) as sf:
        analysis = sf.analyse_position(BACK_RANK_MATE_IN_1_FEN)
        assert analysis["best_move_uci"] == "a1a8"
        assert analysis["topk"][0]["is_mate"] is True
        assert analysis["topk"][0]["mate_in"] == 1
        # verificacion independiente: el movimiento propuesto es objetivamente mate
        board = chess.Board(BACK_RANK_MATE_IN_1_FEN)
        board.push(chess.Move.from_uci(analysis["best_move_uci"]))
        assert board.is_checkmate()


@requires_stockfish
def test_multipv_returns_k_candidates_ranked():
    with StockfishEnricher(STOCKFISH_PATH, depth=6, multipv=3) as sf:
        analysis = sf.analyse_position(chess.STARTING_FEN)
        assert len(analysis["topk"]) <= 3
        assert len(analysis["topk"]) >= 1
        ranks = [c["rank"] for c in analysis["topk"]]
        assert ranks == sorted(ranks)
        # el score del primer candidato nunca es peor (para quien mueve) que el segundo
        if len(analysis["topk"]) >= 2:
            assert analysis["topk"][0]["score_cp"] >= analysis["topk"][1]["score_cp"]


@requires_stockfish
def test_score_perspective_is_side_to_move():
    """El mismo desbalance material debe evaluarse con signo opuesto segun
    quien tiene el turno (perspectiva side-to-move, nunca blanca fija)."""
    with StockfishEnricher(STOCKFISH_PATH, depth=6, multipv=1) as sf:
        # blancas con una torre de mas, blancas al turno -> eval fuertemente positiva
        fen_white_up_white_to_move = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
        a1 = sf.analyse_position(fen_white_up_white_to_move)
        assert a1["best_eval_cp"] > 100

        # blancas con una torre de mas, pero NEGRAS al turno -> eval fuertemente
        # negativa desde la perspectiva de quien mueve (esta perdiendo)
        fen_white_up_black_to_move = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
        a2 = sf.analyse_position(fen_white_up_black_to_move)
        assert a2["best_eval_cp"] < -100


@requires_stockfish
def test_centipawn_loss_zero_for_engine_best_move():
    with StockfishEnricher(STOCKFISH_PATH, depth=8, multipv=1) as sf:
        result = sf.evaluate_human_move(BACK_RANK_MATE_IN_1_FEN, "a1a8", QUALITY_THRESHOLDS)
        assert result["centipawn_loss"] == 0
        assert result["move_quality"] == "best"


@requires_stockfish
def test_centipawn_loss_positive_for_blunder():
    with StockfishEnricher(STOCKFISH_PATH, depth=8, multipv=1) as sf:
        # en vez de dar mate en 1 (Ra8#), blancas mueven el rey -- desperdicia el mate
        result = sf.evaluate_human_move(BACK_RANK_MATE_IN_1_FEN, "g1h1", QUALITY_THRESHOLDS)
        assert result["centipawn_loss"] > 0
        assert result["move_quality"] in {"inaccuracy", "mistake", "blunder"}


def test_classify_move_quality_thresholds_configurable():
    assert classify_move_quality(0, QUALITY_THRESHOLDS) == "best"
    assert classify_move_quality(5, QUALITY_THRESHOLDS) == "best"
    assert classify_move_quality(15, QUALITY_THRESHOLDS) == "excellent"
    assert classify_move_quality(500, QUALITY_THRESHOLDS) == "blunder"
    strict = {"best": 0, "excellent": 5, "good": 10, "inaccuracy": 20, "mistake": 40}
    assert classify_move_quality(6, strict) == "good"


def test_softmax_policy_from_topk_sums_to_one_and_prefers_higher_score():
    topk = [{"move_id": 1, "score_cp": 100}, {"move_id": 2, "score_cp": 0}, {"move_id": 3, "score_cp": -50}]
    policy = softmax_policy_from_topk(topk, temperature_cp=150.0)
    total = sum(c["weight"] for c in policy)
    assert abs(total - 1.0) < 1e-6
    assert policy[0]["weight"] > policy[1]["weight"] > policy[2]["weight"]


def test_softmax_policy_empty_topk_returns_empty():
    assert softmax_policy_from_topk([], temperature_cp=150.0) == []
