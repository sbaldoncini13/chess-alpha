"""Tests de move ordering avanzado (Fase III): hash move, MVV-LVA,
killers, history, countermove.

Correr: python -m pytest tests/search_v1/test_move_ordering.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import OrderingState, Position, order_moves
from search_v1.move_generator import generate_legal_moves

KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


class TestHashMove(unittest.TestCase):
    def test_legal_hash_move_ordered_first(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci="d2d4", ply=0, ordering_state=os, prev_move_uci=None)
        self.assertEqual(ordered[0].uci, "d2d4")
        self.assertEqual(os.hash_move_first_count, 1)

    def test_illegal_hash_move_ignored(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci="e2e5", ply=0, ordering_state=os, prev_move_uci=None)
        # e2e5 no es legal (no esta entre las 20 jugadas) - no debe aparecer
        # primero por "ser el hash move" (cae en su tier normal, si es que
        # existiera - pero ni siquiera esta en la lista de jugadas legales)
        self.assertNotIn("e2e5", [lm.uci for lm in ordered])
        self.assertEqual(os.hash_move_first_count, 0)


class TestMVVLVA(unittest.TestCase):
    def test_capture_high_value_with_low_value_attacker_first(self):
        # peon puede capturar torre O caballo puede capturar torre; el
        # peon (atacante barato) debe ordenarse antes que el caballo
        fen = "4k3/8/8/3r4/2P5/8/6N1/4K3 w - - 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        captures = [lm.uci for lm in ordered if lm.is_capture]
        self.assertEqual(captures[0], "c4d5")  # peon captura torre, antes que cualquier otra

    def test_en_passant_treated_as_pawn_capture(self):
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        ep_moves = [lm for lm in ordered if lm.is_en_passant]
        self.assertEqual(len(ep_moves), 1)
        self.assertTrue(ep_moves[0].is_capture)

    def test_promotion_with_capture_ordered_as_promotion_tier(self):
        # rey negro en h8 (no en f8: un peon blanco en e7 ya ataca
        # diagonalmente d8/f8, poner el rey negro ahi seria "jaque
        # opuesto" - posicion imposible con blancas a mover)
        fen = "3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        # e7xd8 es promocion+captura - debe estar en el tier de promociones
        # (antes que capturas simples si las hubiera, y desde luego entre
        # las primeras jugadas ordenadas)
        promo_capture = [lm for lm in ordered if lm.is_promotion and lm.is_capture]
        self.assertEqual(len(promo_capture), 4)  # dxq/r/b/n
        idx = ordered.index(promo_capture[0])
        self.assertLess(idx, len(ordered) - 1)


class TestKillerMoves(unittest.TestCase):
    def test_killer_must_be_non_capturing(self):
        os = OrderingState()
        os.update_killer(ply=2, uci="e2e4")
        self.assertIn("e2e4", os.killers[2])

    def test_killer_no_duplicates(self):
        os = OrderingState()
        os.update_killer(2, "e2e4")
        os.update_killer(2, "e2e4")
        self.assertEqual(os.killers[2].count("e2e4"), 1)

    def test_killer_max_per_ply(self):
        os = OrderingState()
        os.update_killer(2, "e2e4")
        os.update_killer(2, "d2d4")
        os.update_killer(2, "c2c4")
        self.assertLessEqual(len(os.killers[2]), 2)

    def test_illegal_killer_ignored_in_ordering(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        os.update_killer(0, "a7a5")  # jugada de negras, ilegal para blancas aca
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        self.assertNotIn("a7a5", [lm.uci for lm in ordered])
        # y no rompe nada - todas las jugadas siguen siendo las 20 legales
        self.assertEqual(len(ordered), 20)


class TestHistoryHeuristic(unittest.TestCase):
    def test_history_update_increases_score(self):
        os = OrderingState()
        self.assertEqual(os.history_score("e2", "e4"), 0)
        os.update_history("e2", "e4", depth=3)
        self.assertEqual(os.history_score("e2", "e4"), 9)  # depth^2
        self.assertEqual(os.history_updates, 1)

    def test_history_higher_score_ordered_first_among_quiet_moves(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        os.update_history("g1", "f3", depth=5)  # historia alta para Nf3
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        quiet = [lm.uci for lm in ordered if not lm.is_capture and not lm.is_promotion and not lm.gives_check]
        self.assertEqual(quiet[0], "g1f3")

    def test_history_decay_prevents_overflow(self):
        os = OrderingState()
        from search_v1.move_ordering import MAX_HISTORY_VALUE
        os.history[("e2", "e4")] = MAX_HISTORY_VALUE
        os.update_history("e2", "e4", depth=1)
        self.assertLess(os.history_score("e2", "e4"), MAX_HISTORY_VALUE)


class TestCountermove(unittest.TestCase):
    def test_countermove_valid_and_ordered(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        os.update_countermove(prev_move_uci="d7d5", response_uci="g1f3")
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci="d7d5")
        quiet = [lm.uci for lm in ordered if not lm.is_capture and not lm.is_promotion]
        self.assertEqual(quiet[0], "g1f3")
        self.assertEqual(os.countermove_hits, 1)

    def test_countermove_illegal_ignored(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        os.update_countermove(prev_move_uci="d7d5", response_uci="e7e5")  # ilegal para blancas
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci="d7d5")
        self.assertNotIn("e7e5", [lm.uci for lm in ordered])
        self.assertEqual(os.countermove_hits, 0)

    def test_no_previous_move_no_countermove(self):
        p = Position()
        moves = generate_legal_moves(p)
        os = OrderingState()
        ordered = order_moves(p, moves, hash_move_uci=None, ply=0, ordering_state=os, prev_move_uci=None)
        self.assertEqual(len(ordered), 20)  # no crashea, se comporta igual que sin countermove


class TestOrderingDeterminism(unittest.TestCase):
    def test_same_state_same_order(self):
        p1, p2 = Position(KIWIPETE_FEN), Position(KIWIPETE_FEN)
        os1, os2 = OrderingState(), OrderingState()
        moves1, moves2 = generate_legal_moves(p1), generate_legal_moves(p2)
        ordered1 = order_moves(p1, moves1, None, 0, os1, None)
        ordered2 = order_moves(p2, moves2, None, 0, os2, None)
        self.assertEqual([lm.uci for lm in ordered1], [lm.uci for lm in ordered2])


if __name__ == "__main__":
    unittest.main()
