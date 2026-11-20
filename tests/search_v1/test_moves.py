"""Tests de aplicacion de movimientos: tipos (captura, promocion, enroque,
al paso), banderas, y aplicacion sobre el tablero.

Correr: python -m pytest tests/search_v1/test_moves.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import Move, Position, generate_legal_moves


class TestMoveTypes(unittest.TestCase):
    def test_simple_pawn_move(self):
        p = Position()
        m = Move.from_uci("e2e4")
        self.assertTrue(p.is_legal(m))
        self.assertFalse(p.is_capture(m))
        self.assertFalse(m.is_promotion)

    def test_capture(self):
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        p = Position(fen)
        m = Move.from_uci("e4d5")
        self.assertTrue(p.is_legal(m))
        self.assertTrue(p.is_capture(m))

    def test_promotion_to_each_piece(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        p = Position(fen)
        for letter, expected in [("q", "q"), ("r", "r"), ("b", "b"), ("n", "n")]:
            m = Move.from_uci(f"a7a8{letter}")
            self.assertTrue(p.is_legal(m), f"a7a8{letter} deberia ser legal")
            self.assertTrue(m.is_promotion)
            self.assertEqual(m.promotion_piece, expected)

    def test_castling_kingside_allowed(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        p = Position(fen)
        m = Move.from_uci("e1g1")
        self.assertTrue(p.is_legal(m))
        self.assertTrue(p.is_castling(m))

    def test_castling_forbidden_when_rights_lost(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1"  # blancas perdieron KQ
        p = Position(fen)
        m = Move.from_uci("e1g1")
        self.assertFalse(p.is_legal(m))

    def test_castling_forbidden_through_check(self):
        # torre negra en f8 ataca la casilla de paso f1 (rey blanco pasa
        # por f1 al enrocar corto e1g1)
        fen = "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1"
        p = Position(fen)
        m = Move.from_uci("e1g1")
        self.assertFalse(p.is_legal(m))

    def test_en_passant_allowed(self):
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        p = Position(fen)
        m = Move.from_uci("e5d6")
        self.assertTrue(p.is_legal(m))
        self.assertTrue(p.is_en_passant(m))
        self.assertTrue(p.is_capture(m))

    def test_en_passant_expired_not_offered(self):
        # mismo patron de piezas pero sin el en-passant en el FEN (ya
        # "expiro" - no se jugo inmediatamente despues del avance doble)
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
        p = Position(fen)
        m = Move.from_uci("e5d6")
        self.assertFalse(p.is_legal(m))

    def test_gives_check(self):
        fen = "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1"
        p = Position(fen)
        m = Move.from_uci("f1f8")
        self.assertTrue(p.gives_check(m))


class TestMoveLegalityDeniesPseudoLegal(unittest.TestCase):
    def test_pinned_piece_cannot_move_exposing_king(self):
        # torre blanca en e2, rey blanco en e1, torre negra en e8, rey
        # negro en a8: mover la torre blanca lateralmente expondria al
        # rey - ilegal
        fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
        p = Position(fen)
        m = Move.from_uci("e2a2")
        self.assertFalse(p.is_legal(m))
        # pero SI puede capturar/avanzar en la misma columna (sigue bloqueando)
        m2 = Move.from_uci("e2e3")
        self.assertTrue(p.is_legal(m2))

    def test_must_resolve_existing_check(self):
        # rey blanco en e1 en jaque por torre negra sin defensa en e2:
        # unicas respuestas legales son escapar a d1/f1 (fuera de la fila/
        # columna atacada) o capturar la torre (e1e2, no defendida) -
        # d2/f2 siguen en la fila 2 atacada por la torre, ilegales.
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        legal = {lm.uci for lm in generate_legal_moves(p)}
        self.assertEqual(legal, {"e1d1", "e1f1", "e1e2"})

    def test_discovered_check_flagged(self):
        # alfil blanco se aparta y descubre jaque de la torre en la misma fila
        fen = "4k3/8/8/8/8/8/8/R2BK3 w - - 0 1"
        p = Position(fen)
        m = Move.from_uci("d1h5")
        self.assertTrue(p.is_legal(m))
        self.assertTrue(p.gives_check(m))

    def test_double_check_only_king_moves_legal(self):
        # Doble jaque real, verificado a mano: caballo blanco en d6 ataca
        # e8 (salto d6->e8), torre blanca en e1 ataca e8 por columna e
        # despejada. Rey negro en e8 NO puede capturar ninguna de las dos
        # piezas (ninguna es adyacente) ni bloquear dos lineas de jaque
        # distintas con una sola jugada - unica respuesta posible: mover
        # el rey a una casilla que escape de AMBAS lineas de ataque
        # (d7/d8/f8 confirmadas a mano; e7 sigue en la columna e, f7 sigue
        # atacada por el caballo).
        fen = "4k3/8/3N4/8/8/8/8/K3R3 b - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        legal = {lm.uci for lm in generate_legal_moves(p)}
        self.assertEqual(legal, {"e8d7", "e8d8", "e8f8"})


class TestApplyUndo(unittest.TestCase):
    def test_apply_illegal_raises(self):
        from search_v1 import IllegalMoveError
        p = Position()
        with self.assertRaises(IllegalMoveError):
            p.apply(Move.from_uci("e2e5"))

    def test_apply_updates_board(self):
        p = Position()
        p.apply(Move.from_uci("e2e4"))
        self.assertEqual(p.side_to_move, "black")
        self.assertEqual(p.board.piece_at(chess.E4), chess.Piece(chess.PAWN, chess.WHITE))
        self.assertIsNone(p.board.piece_at(chess.E2))

    def test_undo_without_apply_raises(self):
        from search_v1 import UndoStackEmptyError
        p = Position()
        with self.assertRaises(UndoStackEmptyError):
            p.undo()


if __name__ == "__main__":
    unittest.main()
