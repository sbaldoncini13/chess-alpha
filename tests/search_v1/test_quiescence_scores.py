"""Tests de perspectiva, mate scores, PV y apply/undo dentro de
quiescence (Fase IV, items 24-28).

Correr: python -m pytest tests/search_v1/test_quiescence_scores.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, QuiescenceLimits, SearchLimits, iterative_deepening_search,
    is_mate_score, validate_principal_variation,
)

QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)


class TestScorePerspective(unittest.TestCase):
    def test_score_favors_side_to_move_with_material_edge(self):
        # blancas con una dama de mas (posicion quieta, sin capturas
        # disponibles) - mismo patron ya validado en Fase II, aca se
        # verifica que quiescence no altera la perspectiva
        fen_white_up = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
        res_white = iterative_deepening_search(Position(fen_white_up), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        self.assertGreater(res_white.score, 0)

        fen_black_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
        res_black = iterative_deepening_search(Position(fen_black_to_move), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        self.assertLess(res_black.score, 0)


class TestMateScoreCorrect(unittest.TestCase):
    def test_mate_found_through_quiescence_boundary(self):
        # mate en 1 alcanzable justo en el horizonte de quiescence
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        self.assertTrue(is_mate_score(res.score))
        self.assertEqual(res.mate_distance_plies, 1)

    def test_checkmate_within_quiescence_recursion(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        self.assertTrue(is_mate_score(res.score))
        self.assertLess(res.score, 0)


class TestPVLegalAndReproducible(unittest.TestCase):
    def test_pv_legal_with_quiescence(self):
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])

    def test_pv_reproducible_from_root(self):
        from search_v1 import Move
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        working = Position(fen)
        for uci in res.principal_variation:
            working.apply(Move.from_uci(uci))


class TestApplyUndoExactAndRootIntact(unittest.TestCase):
    def test_apply_undo_exact_through_quiescence(self):
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        p = Position(fen)
        iterative_deepening_search(p, SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
        self.assertEqual(p.fen, fen)

    def test_root_intact_with_tt_and_ordering_and_quiescence(self):
        from search_v1 import OrderingState, TranspositionTable
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        p = Position(fen)
        iterative_deepening_search(
            p, SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )
        self.assertEqual(p.fen, fen)


if __name__ == "__main__":
    unittest.main()
