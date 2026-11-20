"""Tests de aspiration windows (Fase V, items 6-10): fail-low, fail-high,
ventana exitosa, ventana ampliada, fallback a ventana completa - probados
con un stub controlado (no dependen de que una posicion de ajedrez real
dispare cada rama, que seria fragil) + un test de integracion real con
iterative_deepening_search.

Correr: python -m pytest tests/search_v1/test_aspiration.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, AspirationConfig, Position, RepetitionTracker, SearchContext,
    SearchLimits, TranspositionTable, iterative_deepening_search, is_mate_score,
    search_with_aspiration,
)


def _make_stub(scores):
    calls = {"n": 0, "windows": []}

    def stub(position, depth, alpha, beta, context, **kwargs):
        calls["windows"].append((alpha, beta))
        idx = min(calls["n"], len(scores) - 1)
        calls["n"] += 1
        return scores[idx], ["e2e4"]

    return stub, calls


class TestAspirationSuccessfulWindow(unittest.TestCase):
    def test_score_inside_window_no_retry(self):
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        stub, calls = _make_stub([100])
        config = AspirationConfig(initial_window_cp=25)
        score, pv = search_with_aspiration(stub, Position(), 3, 100, ctx, config)
        self.assertEqual(score, 100)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(ctx.aspiration_successful_windows, 1)
        self.assertEqual(ctx.aspiration_fail_lows, 0)
        self.assertEqual(ctx.aspiration_fail_highs, 0)
        self.assertEqual(calls["windows"][0], (75, 125))


class TestAspirationFailLow(unittest.TestCase):
    def test_fail_low_then_success(self):
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        stub, calls = _make_stub([70, 60])  # 70<=75 (fail-low), luego 60 dentro de [0,125]
        config = AspirationConfig(initial_window_cp=25, widening_factor=4)
        score, pv = search_with_aspiration(stub, Position(), 3, 100, ctx, config)
        self.assertEqual(score, 60)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(ctx.aspiration_fail_lows, 1)
        self.assertEqual(ctx.aspiration_fail_highs, 0)
        self.assertEqual(ctx.aspiration_researches, 1)
        self.assertEqual(ctx.aspiration_successful_windows, 1)
        self.assertEqual(calls["windows"][0], (75, 125))
        self.assertEqual(calls["windows"][1], (0, 125))  # solo el lado bajo se amplia


class TestAspirationFailHigh(unittest.TestCase):
    def test_fail_high_then_success(self):
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        stub, calls = _make_stub([130, 90])  # 130>=125 (fail-high), luego 90 dentro de [75,200]
        config = AspirationConfig(initial_window_cp=25, widening_factor=4)
        score, pv = search_with_aspiration(stub, Position(), 3, 100, ctx, config)
        self.assertEqual(score, 90)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(ctx.aspiration_fail_highs, 1)
        self.assertEqual(ctx.aspiration_fail_lows, 0)
        self.assertEqual(ctx.aspiration_researches, 1)
        self.assertEqual(calls["windows"][0], (75, 125))
        self.assertEqual(calls["windows"][1], (75, 200))  # solo el lado alto se amplia


class TestAspirationWidening(unittest.TestCase):
    def test_window_widens_progressively_on_repeated_fail_low(self):
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        stub, calls = _make_stub([70, -10, 42])  # 2 fail-lows seguidos, luego exito
        config = AspirationConfig(initial_window_cp=25, widening_factor=4, max_retries=4)
        score, pv = search_with_aspiration(stub, Position(), 3, 100, ctx, config)
        self.assertEqual(score, 42)
        self.assertEqual(ctx.aspiration_fail_lows, 2)
        self.assertEqual(ctx.aspiration_researches, 2)
        # ventana 1: [75,125]  ventana 2: [0,125] (window=100)  ventana 3: [-300,125] (window=400)
        self.assertEqual(calls["windows"][0], (75, 125))
        self.assertEqual(calls["windows"][1], (0, 125))
        self.assertEqual(calls["windows"][2], (-300, 125))


class TestAspirationFallbackFullWindow(unittest.TestCase):
    def test_fallback_after_max_retries_exceeded(self):
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        # 3 fail-lows con max_retries=2 -> agota reintentos -> fallback ventana completa
        stub, calls = _make_stub([70, -10, -500, 42])
        config = AspirationConfig(initial_window_cp=25, widening_factor=4, max_retries=2)
        score, pv = search_with_aspiration(stub, Position(), 3, 100, ctx, config)
        self.assertEqual(score, 42)
        self.assertEqual(calls["n"], 4)
        self.assertEqual(ctx.aspiration_fallbacks, 1)
        self.assertEqual(calls["windows"][-1], (-INFINITY, INFINITY))

    def test_never_returns_partial_result_as_complete(self):
        # con max_nodes muy bajo, la iteracion se aborta y se descarta -
        # aspiration NUNCA debe fingir que una ventana angosta incompleta
        # es el resultado final de la profundidad.
        from search_v1 import StopReason
        p = Position()
        res = iterative_deepening_search(
            p, SearchLimits(max_depth=6, max_nodes=25), use_pvs=True, aspiration=AspirationConfig(),
        )
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertLess(res.completed_depth, 6)


class TestAspirationDoesNotAlterMateScores(unittest.TestCase):
    def test_mate_score_identical_with_and_without_aspiration(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        res_plain = iterative_deepening_search(Position(fen), SearchLimits(max_depth=4), use_pvs=True)
        res_asp = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=4), use_pvs=True, aspiration=AspirationConfig(),
        )
        self.assertEqual(res_plain.score, res_asp.score)
        self.assertTrue(is_mate_score(res_asp.score))
        self.assertEqual(res_plain.mate_distance_plies, res_asp.mate_distance_plies)


class TestAspirationIntegration(unittest.TestCase):
    def test_pvs_with_aspiration_matches_pvs_without(self):
        import chess
        fens = [
            chess.STARTING_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ]
        for fen in fens:
            with self.subTest(fen=fen):
                p1 = Position(fen)
                res_plain = iterative_deepening_search(
                    p1, SearchLimits(max_depth=3), tt=TranspositionTable(8192), use_pvs=True,
                )
                p2 = Position(fen)
                res_asp = iterative_deepening_search(
                    p2, SearchLimits(max_depth=3), tt=TranspositionTable(8192), use_pvs=True,
                    aspiration=AspirationConfig(),
                )
                self.assertEqual(res_plain.score, res_asp.score)
                self.assertEqual(p1.fen, fen)
                self.assertEqual(p2.fen, fen)


if __name__ == "__main__":
    unittest.main()
