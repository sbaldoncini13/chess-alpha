"""Tests de integracion PVS + quiescence + iterative deepening (Fase V,
items 12/13/18/19): quiescence en las hojas de PVS, PVS orquestado desde
iterative_deepening_search, PV legal, posicion raiz intacta.

Correr: python -m pytest tests/search_v1/test_pvs_integration.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    OrderingState, Position, QuiescenceLimits, SearchLimits, TranspositionTable,
    iterative_deepening_search, validate_principal_variation,
)

QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)


class TestQuiescenceUnderPVS(unittest.TestCase):
    def test_quiescence_extends_pvs_leaves(self):
        # trampa clasica de horizonte - sin quiescence, depth=1 preferiria
        # la captura que se recaptura y pierde la dama
        fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
        res_no_q = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), use_pvs=True)
        res_q = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS, use_pvs=True)
        self.assertNotEqual(res_no_q.score, res_q.score)
        self.assertIn("q_nodes", res_q.metadata)
        self.assertGreater(res_q.metadata["q_nodes"], 0)

    def test_quiescence_metadata_present_under_pvs(self):
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        res = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS, use_pvs=True,
        )
        self.assertIn("q_max_depth_reached", res.metadata)


class TestIterativeDeepeningWithPVS(unittest.TestCase):
    def test_reaches_requested_depth(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=3), use_pvs=True)
        self.assertEqual(res.completed_depth, 3)

    def test_pvs_telemetry_present(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=3), use_pvs=True)
        for key in ("pvs_null_window_searches", "pvs_researches", "pvs_fail_highs",
                    "pvs_fail_lows", "pvs_full_window_searches"):
            self.assertIn(key, res.metadata)

    def test_pvs_with_full_stack_reaches_depth(self):
        p = Position()
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=3), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(), use_pvs=True,
        )
        self.assertEqual(res.completed_depth, 3)

    def test_negamax_rejects_pvs(self):
        p = Position()
        with self.assertRaises(ValueError):
            iterative_deepening_search(p, SearchLimits(max_depth=2), algorithm="negamax", use_pvs=True)

    def test_pruning_config_requires_use_pvs(self):
        from search_v1 import PruningConfig
        p = Position()
        with self.assertRaises(ValueError):
            iterative_deepening_search(p, SearchLimits(max_depth=2), pruning_config=PruningConfig(null_move_enabled=True))


class TestPVLegalAndRootIntact(unittest.TestCase):
    def test_pv_legal_full_stack(self):
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        res = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=3), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(), use_pvs=True,
        )
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])

    def test_root_intact_full_stack(self):
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        p = Position(fen)
        iterative_deepening_search(
            p, SearchLimits(max_depth=3), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(), use_pvs=True,
        )
        self.assertEqual(p.fen, fen)


if __name__ == "__main__":
    unittest.main()
