"""Tests de Negamax de referencia (Fase II): profundidad 0-3, terminales
dentro de la busqueda, determinismo, no mutacion de la posicion raiz.

Correr: python -m pytest tests/search_v1/test_negamax.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    Position, RepetitionTracker, SearchContext, SearchLimits,
    evaluate_static_relative, negamax,
)

MATE_IN_1_FEN = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"


def _ctx(max_depth=5):
    return SearchContext(SearchLimits(max_depth=max_depth), RepetitionTracker())


class TestNegamaxDepth0(unittest.TestCase):
    def test_depth_0_equals_static_eval(self):
        p = Position()
        score, pv = negamax(p, 0, _ctx())
        self.assertEqual(score, evaluate_static_relative(p))
        self.assertEqual(pv, [])

    def test_depth_0_no_moves_searched(self):
        p = Position(MATE_IN_1_FEN)
        ctx = _ctx()
        negamax(p, 0, ctx)
        self.assertEqual(ctx.nodes, 1)
        self.assertEqual(ctx.leaf_nodes, 1)


class TestNegamaxDepth1to3(unittest.TestCase):
    def test_depth_1_finds_mate_in_1(self):
        p = Position(MATE_IN_1_FEN)
        score, pv = negamax(p, 1, _ctx())
        self.assertTrue(score > 900000)  # es un mate score, ver test_mate_scores.py
        self.assertEqual(len(pv), 1)

    def test_depth_2_starting_position_no_crash(self):
        p = Position()
        score, pv = negamax(p, 2, _ctx())
        self.assertIsInstance(score, int)
        self.assertEqual(len(pv), 2)

    def test_depth_3_finds_mate_in_2(self):
        p = Position("k7/8/2K5/8/8/8/8/7Q w - - 0 1")
        score, pv = negamax(p, 3, _ctx())
        self.assertEqual(pv, ["c6b6", "a8b8", "h1b7"])
        self.assertEqual(score, 999997)


class TestNegamaxTerminalsWithinSearch(unittest.TestCase):
    def test_checkmate_at_root_depth_gt_0(self):
        # rey negro ya en jaque mate - detect_terminal se consulta ANTES
        # de generar jugadas, incluso con depth>0
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        p = Position(fen)
        ctx = _ctx()
        score, pv = negamax(p, 3, ctx)
        self.assertEqual(pv, [])
        self.assertEqual(ctx.terminal_nodes, 1)
        self.assertEqual(ctx.nodes, 1)

    def test_stalemate_within_search(self):
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        p = Position(fen)
        score, pv = negamax(p, 2, _ctx())
        self.assertEqual(score, 0)
        self.assertEqual(pv, [])

    def test_insufficient_material_within_search(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
        p = Position(fen)
        score, pv = negamax(p, 3, _ctx())
        self.assertEqual(score, 0)

    def test_fifty_move_rule_within_search(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 100 60"
        p = Position(fen)
        score, pv = negamax(p, 2, _ctx())
        self.assertEqual(score, 0)
        self.assertEqual(pv, [])

    def test_repetition_within_search_line(self):
        # Pre-sembramos el RepetitionTracker simulando que la posicion
        # resultante de 1.Nf3 YA ocurrio dos veces antes en la partida
        # real (fuera de esta busqueda, historial previo) - al explorar la
        # jugada g1f3 DENTRO de la busqueda, esa posicion pasa a ser la
        # 3ra ocurrencia y debe detectarse como tablas por repeticion
        # (score 0, terminal_nodes+=1) sin necesidad de profundidad extra
        # para formarla dentro del arbol de busqueda (siembra legitima
        # para probar el conteo del tracker de forma barata).
        from search_v1 import Move
        p = Position()
        after_nf3 = p.copy()
        after_nf3.apply(Move.from_uci("g1f3"))

        rep = RepetitionTracker()
        rep.push(p)          # posicion raiz: 1ra ocurrencia (la actual)
        rep.push(after_nf3)  # "tras Nf3": 1ra ocurrencia previa simulada
        rep.push(after_nf3)  # "tras Nf3": 2da ocurrencia previa simulada
        self.assertEqual(rep.count(after_nf3), 2)

        ctx = SearchContext(SearchLimits(max_depth=1), rep)
        negamax(p, 1, ctx)
        # la rama g1f3 debe haber detectado la 3ra ocurrencia como terminal
        self.assertGreaterEqual(ctx.terminal_nodes, 1)
        # el tracker vuelve a su estado previo a la busqueda (push/pop balanceado)
        self.assertEqual(rep.count(after_nf3), 2)


class TestNegamaxDeterminismAndPurity(unittest.TestCase):
    def test_determinism_same_config(self):
        # depth=2 (no depth=3): negamax sin poda en una posicion de
        # mediojuego abierta a depth=3 ya son ~O(30^3) nodos - depth=2
        # alcanza para probar determinismo sin competir de mas por CPU
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p1, p2 = Position(fen), Position(fen)
        s1, pv1 = negamax(p1, 2, _ctx())
        s2, pv2 = negamax(p2, 2, _ctx())
        self.assertEqual(s1, s2)
        self.assertEqual(pv1, pv2)

    def test_root_position_not_mutated(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        negamax(p, 2, _ctx())
        self.assertEqual(p.fen, fen)

    def test_no_pruning_explores_all_moves_at_depth_1(self):
        p = Position()
        ctx = _ctx()
        negamax(p, 1, ctx)
        # 20 jugadas legales en la posicion inicial + 1 nodo raiz = 21 nodos
        self.assertEqual(ctx.nodes, 21)
        self.assertEqual(ctx.alpha_beta_cutoffs, 0)  # negamax nunca poda


if __name__ == "__main__":
    unittest.main()
