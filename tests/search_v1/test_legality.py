"""Tests de legalidad: comparacion contra el oraculo python-chess en
posiciones conocidas (no el dataset completo - eso es la validacion
ligera aparte, ver 129_phase1_dataset_validation.py).

Correr: python -m pytest tests/search_v1/test_legality.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import Position, generate_legal_moves, legal_move_count

# Posicion "Kiwipete" - estandar de la industria para perft/legalidad,
# valor conocido de 48 movimientos legales en el primer nivel (incluye
# enroques en ambos lados, capturas, promociones cercanas, etc.)
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


class TestLegalityAgainstOracle(unittest.TestCase):
    def test_starting_position_count(self):
        p = Position()
        self.assertEqual(legal_move_count(p), 20)

    def test_kiwipete_known_count(self):
        p = Position(KIWIPETE_FEN)
        self.assertEqual(legal_move_count(p), 48)

    def test_matches_python_chess_oracle_on_sample(self):
        sample_fens = [
            chess.STARTING_FEN,
            KIWIPETE_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 4",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # posicion tactica compleja (perft suite)
        ]
        for fen in sample_fens:
            with self.subTest(fen=fen):
                p = Position(fen)
                oracle_board = chess.Board(fen)
                ours = {lm.uci for lm in generate_legal_moves(p)}
                oracle = {m.uci() for m in oracle_board.legal_moves}
                self.assertEqual(ours, oracle)

    def test_move_flags_match_oracle_on_sample(self):
        p = Position(KIWIPETE_FEN)
        oracle_board = chess.Board(KIWIPETE_FEN)
        for lm in generate_legal_moves(p):
            raw = lm.move.raw
            with self.subTest(uci=lm.uci):
                self.assertEqual(lm.is_capture, oracle_board.is_capture(raw))
                self.assertEqual(lm.is_castling, oracle_board.is_castling(raw))
                self.assertEqual(lm.is_en_passant, oracle_board.is_en_passant(raw))
                self.assertEqual(lm.gives_check, oracle_board.gives_check(raw))


if __name__ == "__main__":
    unittest.main()
