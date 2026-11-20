"""Tests de iterative deepening, limites/cancelacion, PV, y busqueda sobre
posiciones con promociones/enroque/al paso (Fase II).

Correr: python -m pytest tests/search_v1/test_iterative_deepening.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    Position, ScoreType, SearchLimits, StopReason,
    iterative_deepening_search, validate_principal_variation,
)

MATE_IN_1_FEN = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
MATE_IN_2_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"


class TestIterativeDeepeningBasic(unittest.TestCase):
    def test_reaches_requested_depth_when_unlimited(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=3))
        self.assertEqual(res.completed_depth, 3)
        self.assertEqual(res.requested_depth, 3)
        self.assertEqual(res.stop_reason, StopReason.DEPTH_COMPLETE)
        self.assertTrue(res.completed)

    def test_root_position_not_mutated(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        iterative_deepening_search(p, SearchLimits(max_depth=3))
        self.assertEqual(p.fen, fen)

    def test_determinism_same_config(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        r1 = iterative_deepening_search(Position(fen), SearchLimits(max_depth=3))
        r2 = iterative_deepening_search(Position(fen), SearchLimits(max_depth=3))
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.best_move, r2.best_move)
        self.assertEqual(r1.principal_variation, r2.principal_variation)

    def test_terminal_root_reports_correctly(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=5))
        self.assertEqual(res.stop_reason, StopReason.TERMINAL_ROOT)
        self.assertIsNone(res.best_move)
        self.assertFalse(res.legal)
        self.assertTrue(res.completed)
        self.assertEqual(res.completed_depth, 0)


class TestMatePreferenceAndDelay(unittest.TestCase):
    def test_finds_mate_in_1(self):
        p = Position(MATE_IN_1_FEN)
        res = iterative_deepening_search(p, SearchLimits(max_depth=3))
        self.assertEqual(res.score_type, ScoreType.MATE)
        self.assertEqual(res.mate_distance_plies, 1)
        self.assertEqual(res.best_move, "h1a8")

    def test_finds_mate_in_2(self):
        p = Position(MATE_IN_2_FEN)
        res = iterative_deepening_search(p, SearchLimits(max_depth=5))
        self.assertEqual(res.score_type, ScoreType.MATE)
        self.assertEqual(res.mate_distance_plies, 3)
        self.assertEqual(res.principal_variation, ["c6b6", "a8b8", "h1b7"])

    def test_delays_inevitable_mate(self):
        # negras (bando perdedor) tienen 3 jugadas legales: b8c8 (mate en
        # 1 mas, distancia total 2 desde aca), b8a8/b8a7 (distancia total
        # 4, maximamente retrasado) - verificado independientemente
        # aplicando cada jugada y buscando la respuesta blanca (ver
        # exploracion previa a este test, valores exactos documentados).
        p = Position("1k6/8/2K5/8/8/8/8/7Q b - - 0 1")
        res = iterative_deepening_search(p, SearchLimits(max_depth=4))
        self.assertEqual(res.mate_distance_plies, -4)
        self.assertIn(res.best_move, ("b8a8", "b8a7"))
        self.assertNotEqual(res.best_move, "b8c8")


class TestLimitsAndCancellation(unittest.TestCase):
    def test_stops_by_depth(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=2))
        self.assertEqual(res.completed_depth, 2)
        self.assertEqual(res.stop_reason, StopReason.DEPTH_COMPLETE)

    def test_stops_by_nodes_conserves_last_complete_iteration(self):
        p = Position()
        # profundidad 1 = 21 nodos exactos (ver test_negamax.py); con
        # max_nodes=25 la iteracion depth=1 completa (21<25) pero depth=2
        # se aborta a mitad de camino - debe conservar el resultado de depth=1
        res = iterative_deepening_search(p, SearchLimits(max_depth=5, max_nodes=25))
        self.assertEqual(res.completed_depth, 1)
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertTrue(res.completed)
        self.assertIsNotNone(res.best_move)

    def test_stops_by_nodes_no_completed_iteration(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=5, max_nodes=1))
        self.assertEqual(res.completed_depth, 0)
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertFalse(res.completed)
        self.assertIsNone(res.best_move)
        self.assertFalse(res.legal)

    def test_stops_by_time(self):
        p = Position()
        # limite de tiempo generosamente chico pero no cero, con
        # max_depth alto para forzar que el tiempo sea la causa real de
        # detencion en alguna profundidad
        res = iterative_deepening_search(p, SearchLimits(max_depth=6, max_time_ms=5, time_check_frequency=1))
        self.assertIn(res.stop_reason, (StopReason.MAX_TIME, StopReason.DEPTH_COMPLETE))
        # completed_depth nunca puede exceder requested_depth
        self.assertLessEqual(res.completed_depth, 6)

    def test_never_returns_partial_iteration_as_definitive(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=5, max_nodes=25))
        # el resultado reportado debe ser EXACTAMENTE el de una busqueda
        # independiente a profundidad completed_depth con los mismos limites de nodos
        p2 = Position()
        from search_v1 import RepetitionTracker, SearchContext, alpha_beta, INFINITY
        ctx = SearchContext(SearchLimits(max_depth=res.completed_depth), RepetitionTracker())
        score, pv = alpha_beta(p2, res.completed_depth, -INFINITY, INFINITY, ctx)
        self.assertEqual(res.score, score)
        self.assertEqual(res.principal_variation, pv)


class TestPrincipalVariationValidator(unittest.TestCase):
    def test_pv_valid_on_normal_search(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=3))
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])

    def test_pv_valid_on_mate_search(self):
        p = Position(MATE_IN_2_FEN)
        res = iterative_deepening_search(p, SearchLimits(max_depth=5))
        problems = validate_principal_variation(Position(MATE_IN_2_FEN), res)
        self.assertEqual(problems, [])

    def test_pv_first_move_matches_best_move(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=3))
        self.assertEqual(res.principal_variation[0], res.best_move)

    def test_pv_reproducible_from_root(self):
        from search_v1 import Move
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=3))
        working = Position(fen)
        for uci in res.principal_variation:
            working.apply(Move.from_uci(uci))  # lanza IllegalMoveError si algo esta mal
        self.assertEqual(len(res.principal_variation), res.completed_depth)


class TestSpecialMovesWithinSearch(unittest.TestCase):
    def test_promotion_chosen_when_clearly_best(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=2))
        self.assertEqual(res.best_move[:4], "a7a8")
        self.assertIn(res.best_move[-1], "qrbn")

    def test_castling_available_and_searchable(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=2))
        self.assertIsNotNone(res.best_move)
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])

    def test_en_passant_searchable(self):
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=2))
        self.assertIsNotNone(res.best_move)
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])


class TestScorePerspective(unittest.TestCase):
    def test_score_favors_side_to_move_with_material_edge(self):
        # blancas a mover con una torre de mas -> score positivo
        fen_white_up = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
        res_white = iterative_deepening_search(Position(fen_white_up), SearchLimits(max_depth=1))
        self.assertGreater(res_white.score, 0)

        # mismo desbalance mirado desde negras a mover -> score negativo
        fen_black_to_move = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
        res_black = iterative_deepening_search(Position(fen_black_to_move), SearchLimits(max_depth=1))
        self.assertLess(res_black.score, 0)


if __name__ == "__main__":
    unittest.main()
