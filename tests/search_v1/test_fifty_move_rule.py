"""Tests de la regla de 50 movimientos (halfmove_clock).

Correr: python -m pytest tests/search_v1/test_fifty_move_rule.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import Move, Position, is_fifty_move_rule


class TestFiftyMoveRule(unittest.TestCase):
    def test_not_available_at_start(self):
        p = Position()
        self.assertFalse(is_fifty_move_rule(p))

    def test_available_at_exactly_100_halfmoves(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 100 60"
        p = Position(fen)
        self.assertEqual(p.halfmove_clock, 100)
        self.assertTrue(is_fifty_move_rule(p))

    def test_not_available_at_99_halfmoves(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 99 60"
        p = Position(fen)
        self.assertFalse(is_fifty_move_rule(p))

    def test_non_capture_non_pawn_move_increments_clock(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 5 60"
        p = Position(fen)
        p.apply(Move.from_uci("e3d3"))
        self.assertEqual(p.halfmove_clock, 6)

    def test_pawn_move_resets_clock(self):
        fen = "8/4P3/4k3/8/8/4K3/8/8 w - - 40 60"
        p = Position(fen)
        p.apply(Move.from_uci("e7e8q"))
        self.assertEqual(p.halfmove_clock, 0)

    def test_capture_resets_clock(self):
        # rey negro en a8 (no en la columna e, para no quedar en jaque de
        # la torre blanca en e5 mientras es turno de blancas)
        fen = "k7/8/8/3pR3/8/8/8/4K3 w - - 40 60"
        p = Position(fen)
        p.apply(Move.from_uci("e5d5"))
        self.assertEqual(p.halfmove_clock, 0)

    def test_clock_reaches_rule_through_play(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 98 60"
        p = Position(fen)
        self.assertFalse(is_fifty_move_rule(p))
        p.apply(Move.from_uci("e3d3"))
        self.assertFalse(is_fifty_move_rule(p))
        p.apply(Move.from_uci("e6d6"))
        self.assertTrue(is_fifty_move_rule(p))


if __name__ == "__main__":
    unittest.main()
