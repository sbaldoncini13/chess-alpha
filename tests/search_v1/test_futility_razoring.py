"""Tests de futility pruning y razoring (Fase V, items 33-34) - ambas
tecnicas quedan DISENADAS pero `enabled=False` por defecto y con estado
DESIGNED_NOT_ENABLED en el gate (ver pvs.py y pruning_config.py: "la
evaluacion estatica actual, material puro, no da margen de seguridad
suficiente" - no se fuerza su aprobacion). Estos tests verifican que las
guardas de seguridad funcionan y que el default desactivado no tiene
ningun efecto - NO son la validacion completa de categorias criticas que
si se exige para null move/LMR.

Correr: python -m pytest tests/search_v1/test_futility_razoring.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, PruningConfig, RepetitionTracker, SearchContext,
    SearchLimits, iterative_deepening_search, pvs, validate_principal_variation,
)


class TestDisabledByDefaultHasNoEffect(unittest.TestCase):
    def test_default_pruning_config_matches_no_pruning_config(self):
        p1 = Position()
        res_none = iterative_deepening_search(p1, SearchLimits(max_depth=4), use_pvs=True)
        p2 = Position()
        res_default = iterative_deepening_search(p2, SearchLimits(max_depth=4), use_pvs=True, pruning_config=PruningConfig())
        self.assertEqual(res_none.score, res_default.score)
        self.assertEqual(res_none.nodes, res_default.nodes)


class TestFutilityDisabledInTacticalSituations(unittest.TestCase):
    def test_never_fires_when_node_in_check(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        pc = PruningConfig(futility_enabled=True, futility_max_depth=3, futility_margin_cp=50)
        pvs(p, 3, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.futility_prunes, 0)

    def test_never_fires_near_mate_score_window(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"  # mate en 2
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        pc = PruningConfig(futility_enabled=True, futility_max_depth=3, futility_margin_cp=50)
        from search_v1 import MATE_SCORE
        # ventana deliberadamente cerca de un score de mate
        pvs(p, 3, MATE_SCORE - 100, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.futility_prunes, 0)

    def test_never_leaves_a_node_without_any_move_searched(self):
        # aun con margenes agresivos, futility nunca debe dejar un nodo
        # sin al menos una jugada buscada (la primera jugada del nodo
        # nunca se salta - garantiza PV/best_move siempre validos).
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=4), use_pvs=True,
            pruning_config=PruningConfig(futility_enabled=True, futility_max_depth=4, futility_margin_cp=1),
        )
        self.assertTrue(res.legal)
        self.assertIsNotNone(res.best_move)
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])
        self.assertEqual(p.fen, fen)


class TestRazoringSafeFallback(unittest.TestCase):
    def test_never_fires_when_node_in_check(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        pc = PruningConfig(razoring_enabled=True, razoring_max_depth=2, razoring_margin_cp=50)
        pvs(p, 2, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.razoring_prunes, 0)

    def test_falls_back_to_quiescence_when_available(self):
        from search_v1 import QuiescenceLimits
        fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
        p = Position(fen)
        qlimits = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=1), quiescence_limits=qlimits, use_pvs=True,
            pruning_config=PruningConfig(razoring_enabled=True, razoring_max_depth=1, razoring_margin_cp=1),
        )
        self.assertTrue(res.legal)
        self.assertEqual(p.fen, fen)

    def test_root_intact_and_legal_result(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=3), use_pvs=True,
            pruning_config=PruningConfig(razoring_enabled=True, razoring_max_depth=1, razoring_margin_cp=1),
        )
        self.assertTrue(res.legal)
        self.assertEqual(p.fen, fen)


if __name__ == "__main__":
    unittest.main()
