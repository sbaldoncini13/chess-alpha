"""Tests de Position: carga/validacion de FEN, campos derivados, copy().

Correr: python -m pytest tests/search_v1/test_position.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import InvalidFENError, Move, Position

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestPositionValid(unittest.TestCase):
    def test_default_is_starting_position(self):
        p = Position()
        self.assertEqual(p.fen, STARTING_FEN)

    def test_from_fen_roundtrip(self):
        p = Position.from_fen(STARTING_FEN)
        self.assertEqual(p.fen, STARTING_FEN)

    def test_side_to_move_white(self):
        p = Position(STARTING_FEN)
        self.assertEqual(p.side_to_move, "white")

    def test_side_to_move_black(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        p = Position(fen)
        self.assertEqual(p.side_to_move, "black")

    def test_castling_rights_full(self):
        p = Position(STARTING_FEN)
        self.assertEqual(p.castling_rights, "KQkq")

    def test_castling_rights_none(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        p = Position(fen)
        self.assertEqual(p.castling_rights, "-")

    def test_en_passant_none(self):
        p = Position(STARTING_FEN)
        self.assertIsNone(p.en_passant_square)

    def test_en_passant_present(self):
        # peon negro recien avanzado a d5, peon blanco en e5 SI puede
        # capturar al paso a d6 - es un caso donde la casilla al paso es
        # realmente relevante (a diferencia de una simplemente listada sin
        # ningun peon en condiciones de capturarla, que python-chess omite
        # de fen() con en_passant='legal', el default)
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        p = Position(fen)
        self.assertEqual(p.en_passant_square, "d6")

    def test_halfmove_clock(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 1"
        p = Position(fen)
        self.assertEqual(p.halfmove_clock, 7)

    def test_fullmove_number(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 42"
        p = Position(fen)
        self.assertEqual(p.fullmove_number, 42)

    def test_is_check_false_at_start(self):
        p = Position()
        self.assertFalse(p.is_check)

    def test_is_check_true(self):
        # Fool's mate final position (negras acaban de dar jaque mate,
        # pero para is_check basta cualquier posicion en jaque)
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        p = Position(fen)
        self.assertTrue(p.is_check)

    def test_position_id_matches_known_polyglot_value(self):
        # Valor canonico documentado en la especificacion de Polyglot para
        # la posicion inicial - sirve de ancla independiente ademas de la
        # continuidad con identity.py::position_id.
        p = Position()
        self.assertEqual(p.position_id, "463b96181691fc9c")

    def test_copy_is_independent(self):
        p1 = Position()
        p2 = p1.copy()
        p2.apply(Move.from_uci("e2e4"))
        self.assertEqual(p1.fen, STARTING_FEN)
        self.assertNotEqual(p2.fen, STARTING_FEN)

    def test_equality_by_full_fen(self):
        p1 = Position(STARTING_FEN)
        p2 = Position(STARTING_FEN)
        self.assertEqual(p1, p2)


class TestPositionInvalidFEN(unittest.TestCase):
    def test_syntactically_malformed(self):
        with self.assertRaises(InvalidFENError):
            Position("this is not a fen")

    def test_wrong_number_of_ranks(self):
        with self.assertRaises(InvalidFENError):
            Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")

    def test_no_kings(self):
        with self.assertRaises(InvalidFENError):
            Position("8/8/8/8/8/8/8/8 w - - 0 1")

    def test_too_many_kings(self):
        with self.assertRaises(InvalidFENError):
            Position("kkkkkkkk/8/8/8/8/8/8/KKKKKKKK w - - 0 1")

    def test_pawns_on_back_rank(self):
        with self.assertRaises(InvalidFENError):
            Position("Pnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_opposite_side_in_check(self):
        # Blancas a mover, pero es negro quien esta "en jaque" imposible:
        # rey negro en e4 atacado por peon blanco con negras a mover NO es
        # invalido por si solo; se prueba en cambio el caso clasico
        # documentado por python-chess: ambos reyes adyacentes.
        with self.assertRaises(InvalidFENError):
            Position("8/8/8/3kK3/8/8/8/8 w - - 0 1")


if __name__ == "__main__":
    unittest.main()
