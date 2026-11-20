"""Tests de evaluacion estatica minima (solo balance material).

Correr: python -m pytest tests/search_v1/test_static_evaluation.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import Position, evaluate_static_relative, evaluate_static_white_pov


class TestStaticEvaluation(unittest.TestCase):
    def test_starting_position_is_zero(self):
        p = Position()
        self.assertEqual(evaluate_static_white_pov(p), 0)

    def test_determinism(self):
        p = Position("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        v1 = evaluate_static_white_pov(p)
        v2 = evaluate_static_white_pov(p)
        v3 = evaluate_static_white_pov(p)
        self.assertEqual(v1, v2)
        self.assertEqual(v2, v3)

    def test_white_extra_queen_is_positive(self):
        fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
        p = Position(fen)
        self.assertEqual(evaluate_static_white_pov(p), 900)

    def test_black_extra_queen_is_negative(self):
        fen = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"
        p = Position(fen)
        self.assertEqual(evaluate_static_white_pov(p), -900)

    def test_symmetric_under_color_mirror(self):
        # misma estructura, colores invertidos (dama para el otro bando) ->
        # el score de blancas debe invertir el signo exactamente
        fen_white_up = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
        fen_black_up = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"
        p1 = Position(fen_white_up)
        p2 = Position(fen_black_up)
        self.assertEqual(evaluate_static_white_pov(p1), -evaluate_static_white_pov(p2))

    def test_relative_perspective_flips_for_black(self):
        fen_white_to_move = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
        fen_black_to_move = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
        p_white = Position(fen_white_to_move)
        p_black = Position(fen_black_to_move)
        self.assertEqual(evaluate_static_relative(p_white), 900)
        self.assertEqual(evaluate_static_relative(p_black), -900)

    def test_all_piece_types_valued(self):
        # una de cada pieza extra para blancas (sin rey, ya contado 0)
        fen = "4k3/8/8/8/8/8/8/QRBNK3 w - - 0 1"
        p = Position(fen)
        expected = 900 + 500 + 330 + 320
        self.assertEqual(evaluate_static_white_pov(p), expected)


if __name__ == "__main__":
    unittest.main()
