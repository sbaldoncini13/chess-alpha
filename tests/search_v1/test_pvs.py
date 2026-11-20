"""Tests de PVS (Fase V, items 1-3/4-5/11/25-28 del pedido): igualdad de
score/mate_distance contra Alpha-Beta en profundidad 1-4, null-window sin
y con re-search, bounds EXACT/LOWER/UPPER de TT bajo PVS, promocion/al
paso/enroque, determinismo.

Correr: python -m pytest tests/search_v1/test_pvs.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    INFINITY, Bound, OrderingState, Position, RepetitionTracker, SearchContext,
    SearchLimits, TranspositionTable, alpha_beta, pvs,
)

SAMPLE_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "k7/8/2K5/8/8/8/8/7Q w - - 0 1",  # mate en 2
]


def _run_ab(fen, depth, tt=None, ordering_state=None):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=depth), RepetitionTracker())
    score, pv = alpha_beta(p, depth, -INFINITY, INFINITY, ctx, tt=tt, ordering_state=ordering_state)
    return score, pv, p, ctx


def _run_pvs(fen, depth, tt=None, ordering_state=None, pruning_config=None):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=depth), RepetitionTracker())
    score, pv = pvs(p, depth, -INFINITY, INFINITY, ctx, tt=tt, ordering_state=ordering_state, pruning_config=pruning_config)
    return score, pv, p, ctx


class TestPVSMatchesAlphaBeta(unittest.TestCase):
    """Items 1-3: PVS vs Alpha-Beta profundidad 1-4, igualdad de score y
    mate_distance, sin TT/ordering (oraculo puro) y con TT+ordering."""

    def test_pure_pvs_matches_pure_alpha_beta(self):
        for fen in SAMPLE_FENS:
            for depth in (1, 2, 3, 4):
                with self.subTest(fen=fen, depth=depth):
                    score_ab, _, _, _ = _run_ab(fen, depth)
                    score_pvs, _, _, _ = _run_pvs(fen, depth)
                    self.assertEqual(score_ab, score_pvs)

    def test_pvs_with_tt_and_ordering_matches_alpha_beta(self):
        for fen in SAMPLE_FENS:
            for depth in (1, 2, 3, 4):
                with self.subTest(fen=fen, depth=depth):
                    score_ab, _, _, _ = _run_ab(fen, depth, tt=TranspositionTable(8192), ordering_state=OrderingState())
                    score_pvs, _, _, _ = _run_pvs(fen, depth, tt=TranspositionTable(8192), ordering_state=OrderingState())
                    self.assertEqual(score_ab, score_pvs)

    def test_mate_distance_identical(self):
        from search_v1 import mate_distance_plies_from_score
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        score_ab, _, _, _ = _run_ab(fen, 3)
        score_pvs, _, _, _ = _run_pvs(fen, 3)
        self.assertEqual(
            mate_distance_plies_from_score(score_ab),
            mate_distance_plies_from_score(score_pvs),
        )
        self.assertEqual(mate_distance_plies_from_score(score_pvs), 3)


class TestNullWindow(unittest.TestCase):
    """Items 4-5: null-window sin re-search (movimiento ordenado primero
    ya es el mejor - los demas confirman <=alpha sin re-buscar) y CON
    re-search (una jugada tardia supera alpha, forzando ventana completa)."""

    def test_null_window_searches_without_research_when_ordering_is_good(self):
        # posicion con una jugada dominante clara y buen ordenamiento
        # (TT+ordering) - la mayoria de ventanas nulas deberian confirmar
        # fail-low sin necesitar re-search.
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"  # Rxd4 gana la dama
        _, _, _, ctx = _run_pvs(fen, 3, tt=TranspositionTable(8192), ordering_state=OrderingState())
        self.assertGreater(ctx.pvs_null_window_searches, 0)
        self.assertGreater(ctx.pvs_fail_lows, 0)

    def test_research_happens_and_produces_correct_score(self):
        # sin TT/ordering (orden Fase II, tier fijo) es mucho mas facil que
        # una jugada tardia "sorprenda" y dispare un re-search real -
        # lo importante es que el score final siga siendo correcto.
        found_research = False
        for fen in SAMPLE_FENS:
            score_ab, _, _, _ = _run_ab(fen, 3)
            score_pvs, _, _, ctx = _run_pvs(fen, 3)
            self.assertEqual(score_ab, score_pvs)
            if ctx.pvs_researches > 0:
                found_research = True
        self.assertTrue(found_research, "ninguna posicion de muestra disparo un re-search de PVS a depth=3")

    def test_null_window_buckets_are_exhaustive(self):
        # cada null-window search cae en EXACTAMENTE uno de los 3 buckets
        for fen in SAMPLE_FENS:
            _, _, _, ctx = _run_pvs(fen, 3, tt=TranspositionTable(8192), ordering_state=OrderingState())
            self.assertEqual(
                ctx.pvs_null_window_searches,
                ctx.pvs_fail_lows + ctx.pvs_fail_highs + ctx.pvs_researches,
            )


class TestTTBoundsUnderPVS(unittest.TestCase):
    """Item 11: EXACT/LOWER/UPPER se guardan correctamente bajo PVS."""

    def test_exact_bound_stored(self):
        tt = TranspositionTable(4096)
        p = Position(chess.STARTING_FEN)
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        score, pv = pvs(p, 2, -INFINITY, INFINITY, ctx, tt=tt)
        entry = tt.probe(p.zobrist_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.EXACT)
        self.assertEqual(entry.score, score)

    def test_lower_bound_on_beta_cutoff(self):
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
        tt = TranspositionTable(4096)
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        score, pv = pvs(p, 2, -INFINITY, 50, ctx, tt=tt)
        entry = tt.probe(p.zobrist_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.LOWER_BOUND)

    def test_upper_bound_on_fail_low(self):
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
        tt = TranspositionTable(4096)
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        score, pv = pvs(p, 2, 5000, INFINITY, ctx, tt=tt)
        entry = tt.probe(p.zobrist_key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.UPPER_BOUND)

    def test_tt_hit_reused_second_call_same_score(self):
        fen = chess.STARTING_FEN
        tt = TranspositionTable(8192)
        score1, _, _, _ = _run_pvs(fen, 3, tt=tt)
        score2, _, _, ctx2 = _run_pvs(fen, 3, tt=tt)
        self.assertEqual(score1, score2)
        self.assertGreater(tt.hits, 0)


class TestSpecialMoves(unittest.TestCase):
    """Items 25-27: promocion, captura al paso, enroque - bajo PVS."""

    def test_promotion_found(self):
        fen = "8/P7/8/4k3/8/8/8/4K3 w - - 0 1"  # a7-a8 corona
        score_ab, pv_ab, _, _ = _run_ab(fen, 2)
        score_pvs, pv_pvs, _, _ = _run_pvs(fen, 2)
        self.assertEqual(score_ab, score_pvs)
        self.assertEqual(pv_ab[0], pv_pvs[0])
        self.assertTrue(pv_pvs[0].startswith("a7a8"))

    def test_en_passant_found(self):
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        score_ab, pv_ab, _, _ = _run_ab(fen, 2)
        score_pvs, pv_pvs, _, _ = _run_pvs(fen, 2)
        self.assertEqual(score_ab, score_pvs)

    def test_castling_found(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        score_ab, _, _, _ = _run_ab(fen, 2)
        score_pvs, _, _, _ = _run_pvs(fen, 2)
        self.assertEqual(score_ab, score_pvs)


class TestDeterminism(unittest.TestCase):
    """Item 28: misma posicion/config -> siempre el mismo resultado."""

    def test_repeated_calls_identical(self):
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        results = []
        for _ in range(3):
            score, pv, p, _ = _run_pvs(fen, 3, tt=TranspositionTable(8192), ordering_state=OrderingState())
            results.append((score, tuple(pv)))
            self.assertEqual(p.fen, fen)
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
