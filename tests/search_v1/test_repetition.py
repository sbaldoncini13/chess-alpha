"""Tests de clave de posicion (Zobrist/Polyglot) y deteccion de repeticion.

Correr: python -m pytest tests/search_v1/test_repetition.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import Move, Position, RepetitionTracker, zobrist_key


class TestZobristKeyProperties(unittest.TestCase):
    def test_same_state_same_hash(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p1 = Position(fen)
        p2 = Position(fen)
        self.assertEqual(zobrist_key(p1), zobrist_key(p2))

    def test_different_side_to_move_different_hash(self):
        fen_white = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        fen_black = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        p1 = Position(fen_white)
        p2 = Position(fen_black)
        self.assertNotEqual(zobrist_key(p1), zobrist_key(p2))

    def test_different_castling_rights_different_hash(self):
        fen_full = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        fen_no_castle = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
        p1 = Position(fen_full)
        p2 = Position(fen_no_castle)
        self.assertNotEqual(zobrist_key(p1), zobrist_key(p2))

    def test_relevant_en_passant_different_hash(self):
        fen_with_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        fen_without_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
        p1 = Position(fen_with_ep)
        p2 = Position(fen_without_ep)
        self.assertNotEqual(zobrist_key(p1), zobrist_key(p2))

    def test_halfmove_clock_not_mixed_into_key(self):
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen2 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 37 1"
        p1 = Position(fen1)
        p2 = Position(fen2)
        self.assertEqual(zobrist_key(p1), zobrist_key(p2))

    def test_apply_undo_restores_exact_hash(self):
        p = Position()
        original = zobrist_key(p)
        p.apply(Move.from_uci("e2e4"))
        self.assertNotEqual(zobrist_key(p), original)
        p.undo()
        self.assertEqual(zobrist_key(p), original)


class TestRepetitionTracker(unittest.TestCase):
    def test_threefold_repetition_via_king_shuffle(self):
        p = Position()
        tracker = RepetitionTracker()
        tracker.push(p)  # posicion inicial, ocurrencia 1

        shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]  # vuelve a la posicion inicial
        for _ in range(2):  # dos vueltas completas -> 2 repeticiones extra = 3 total
            for uci in shuffle:
                p.apply(Move.from_uci(uci))
                tracker.push(p)

        self.assertTrue(tracker.is_threefold_repetition())
        self.assertEqual(tracker.count(p), 3)

    def test_not_repetition_before_third_occurrence(self):
        p = Position()
        tracker = RepetitionTracker()
        tracker.push(p)
        shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        for uci in shuffle:
            p.apply(Move.from_uci(uci))
            tracker.push(p)
        # solo 2 ocurrencias de la posicion inicial hasta aca
        self.assertEqual(tracker.count(p), 2)
        self.assertFalse(tracker.is_threefold_repetition())

    def test_push_pop_roundtrip(self):
        p = Position()
        tracker = RepetitionTracker()
        tracker.push(p)
        self.assertEqual(len(tracker), 1)
        tracker.pop()
        self.assertEqual(len(tracker), 0)
        self.assertEqual(tracker.count(p), 0)

    def test_pop_empty_raises(self):
        tracker = RepetitionTracker()
        with self.assertRaises(IndexError):
            tracker.pop()


if __name__ == "__main__":
    unittest.main()
