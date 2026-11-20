"""Tests de limites/cancelacion bajo PVS (Fase V, items 14-17): limite de
nodos, limite de tiempo, cancelacion limpia (SearchAborted descarta la
iteracion parcial), ultima iteracion completa conservada.

Correr: python -m pytest tests/search_v1/test_pvs_limits.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, RepetitionTracker, SearchContext, SearchLimits, SearchAborted,
    StopReason, TranspositionTable, iterative_deepening_search, pvs,
)


class TestNodeLimit(unittest.TestCase):
    def test_interruption_by_nodes_no_partial_result(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=6, max_nodes=30), use_pvs=True)
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertTrue(res.completed)
        self.assertGreaterEqual(res.completed_depth, 1)

    def test_node_limit_raises_search_aborted_directly(self):
        p = Position()
        ctx = SearchContext(SearchLimits(max_depth=10, max_nodes=5), RepetitionTracker())
        with self.assertRaises(SearchAborted):
            pvs(p, 10, -INFINITY, INFINITY, ctx)


class TestTimeLimit(unittest.TestCase):
    def test_interruption_by_time(self):
        p = Position()
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=8, max_time_ms=5, time_check_frequency=1), use_pvs=True,
        )
        self.assertIn(res.stop_reason, (StopReason.MAX_TIME, StopReason.DEPTH_COMPLETE))


class TestCancellationDiscardsPartialIteration(unittest.TestCase):
    def test_partial_iteration_never_returned_as_completed_depth(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=6, max_nodes=30), use_pvs=True)
        # completed_depth debe ser la ULTIMA iteracion que termino sin abortar
        self.assertLess(res.completed_depth, 6)

    def test_partial_iteration_discarded_with_aspiration(self):
        from search_v1 import AspirationConfig
        p = Position()
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=6, max_nodes=40), use_pvs=True, aspiration=AspirationConfig(),
        )
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertTrue(res.completed)


class TestLastCompleteIterationConserved(unittest.TestCase):
    def test_reproducible_independently(self):
        p = Position()
        res = iterative_deepening_search(p, SearchLimits(max_depth=5, max_nodes=30), use_pvs=True)
        p2 = Position()
        ctx2 = SearchContext(SearchLimits(max_depth=res.completed_depth), RepetitionTracker())
        score2, _ = pvs(p2, res.completed_depth, -INFINITY, INFINITY, ctx2)
        self.assertEqual(res.score, score2)


if __name__ == "__main__":
    unittest.main()
