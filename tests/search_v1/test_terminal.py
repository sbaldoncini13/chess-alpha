"""Tests de deteccion de terminales: jaque mate, ahogado, material
insuficiente.

Correr: python -m pytest tests/search_v1/test_terminal.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import Position, TerminalStatus, detect_terminal


class TestCheckmate(unittest.TestCase):
    def test_fools_mate(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        p = Position(fen)
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.CHECKMATE)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.winner, "black")

    def test_scholars_mate(self):
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        p = Position(fen)
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.CHECKMATE)
        self.assertEqual(result.winner, "white")

    def test_not_checkmate_when_escape_exists(self):
        p = Position()
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.NONE)
        self.assertFalse(result.is_terminal)


class TestStalemate(unittest.TestCase):
    def test_classic_stalemate(self):
        # rey negro en a8 sin jugadas legales, sin estar en jaque (rey
        # blanco a7, dama blanca b6) - posicion de ahogado de manual
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        p = Position(fen)
        self.assertFalse(p.is_check)
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.STALEMATE)
        self.assertTrue(result.is_terminal)
        self.assertIsNone(result.winner)


class TestInsufficientMaterial(unittest.TestCase):
    def test_king_vs_king(self):
        p = Position("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.INSUFFICIENT_MATERIAL)

    def test_king_bishop_vs_king(self):
        p = Position("8/8/4k3/8/8/3BK3/8/8 w - - 0 1")
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.INSUFFICIENT_MATERIAL)

    def test_king_knight_vs_king(self):
        p = Position("8/8/4k3/8/8/3NK3/8/8 w - - 0 1")
        result = detect_terminal(p)
        self.assertEqual(result.status, TerminalStatus.INSUFFICIENT_MATERIAL)

    def test_king_rook_vs_king_is_sufficient(self):
        p = Position("8/8/4k3/8/8/3RK3/8/8 w - - 0 1")
        result = detect_terminal(p)
        self.assertNotEqual(result.status, TerminalStatus.INSUFFICIENT_MATERIAL)
        self.assertFalse(result.is_terminal)

    def test_starting_position_is_sufficient(self):
        p = Position()
        result = detect_terminal(p)
        self.assertFalse(result.is_terminal)


if __name__ == "__main__":
    unittest.main()
