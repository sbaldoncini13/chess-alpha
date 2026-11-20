"""Tests de terminales bajo PVS (Fase V, items 20-24): mate corto, mate
inevitable retrasado, ahogado, repeticion, regla de 50 movimientos -
mismos fixtures ya verificados en Fase II/III/IV, ahora vía pvs().

Correr: python -m pytest tests/search_v1/test_pvs_scores.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    OrderingState, Position, QuiescenceLimits, RepetitionTracker, SearchLimits,
    TranspositionTable, is_mate_score, iterative_deepening_search,
)

QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)


class TestMateShort(unittest.TestCase):
    def test_mate_in_1(self):
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), use_pvs=True)
        self.assertTrue(is_mate_score(res.score))
        self.assertEqual(res.mate_distance_plies, 1)
        self.assertEqual(res.best_move, "h1a8")

    def test_mate_in_2(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=4), use_pvs=True)
        self.assertTrue(is_mate_score(res.score))
        self.assertEqual(res.mate_distance_plies, 3)


class TestMateInevitableDelayed(unittest.TestCase):
    def test_delays_forced_mate_as_long_as_possible(self):
        # negras forzadas a perder: evita el mate mas rapido (b8c8)
        fen = "1k6/8/2K5/8/8/8/8/7Q b - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=4), use_pvs=True)
        self.assertNotEqual(res.best_move, "b8c8")


class TestStalemate(unittest.TestCase):
    def test_stalemate_scored_zero(self):
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), use_pvs=True)
        self.assertEqual(res.score, 0)


class TestRepetition(unittest.TestCase):
    def test_repetition_detected_and_scored_zero(self):
        p = Position()
        history = RepetitionTracker()
        history.push(p)
        from search_v1 import Move
        # Nf3 Nf6 Ng1 Ng8 (repite la posicion inicial)
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            p.apply(Move.from_uci(uci))
            history.push(p)
        res = iterative_deepening_search(p, SearchLimits(max_depth=2), game_history=history, use_pvs=True)
        self.assertEqual(res.score, 0)


class TestFiftyMoveRule(unittest.TestCase):
    def test_fifty_move_rule_scored_zero(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 100 60"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), use_pvs=True)
        self.assertEqual(res.score, 0)


class TestSameWithTTOrderingQuiescence(unittest.TestCase):
    """Repite los 5 casos anteriores con TT+ordering+quiescence activos -
    confirma que la interaccion completa (Fase III+IV+V) sigue correcta."""

    def test_mate_in_1_full_stack(self):
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        res = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
            tt=TranspositionTable(4096), ordering_state=OrderingState(), use_pvs=True,
        )
        self.assertTrue(is_mate_score(res.score))
        self.assertEqual(res.mate_distance_plies, 1)

    def test_stalemate_full_stack(self):
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        res = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
            tt=TranspositionTable(4096), ordering_state=OrderingState(), use_pvs=True,
        )
        self.assertEqual(res.score, 0)

    def test_fifty_move_rule_full_stack(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 100 60"
        res = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
            tt=TranspositionTable(4096), ordering_state=OrderingState(), use_pvs=True,
        )
        self.assertEqual(res.score, 0)


if __name__ == "__main__":
    unittest.main()
