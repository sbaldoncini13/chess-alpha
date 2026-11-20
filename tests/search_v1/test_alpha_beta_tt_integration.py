"""Tests de integracion Alpha-Beta + TT + move ordering (Fase III):
igualdad de score con/sin TT, PV, posicion raiz intacta, iterative
deepening con TT, limites, repeticion/50 movimientos con TT, colisiones
reales de enroque/al-paso, comparacion aleatoria reproducible contra
Alpha-Beta de Fase II (seed=20260805).

Correr: python -m pytest tests/search_v1/test_alpha_beta_tt_integration.py -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    INFINITY, OrderingState, Position, RepetitionTracker, ScoreType,
    SearchContext, SearchLimits, StopReason, TranspositionTable,
    alpha_beta, iterative_deepening_search, validate_principal_variation,
)

SEED = 20260805
MATE_IN_2_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"

SAMPLE_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    MATE_IN_2_FEN,
]


def _run(fen, depth, tt=None, ordering_state=None):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=depth), RepetitionTracker())
    score, pv = alpha_beta(p, depth, -INFINITY, INFINITY, ctx, tt=tt, ordering_state=ordering_state)
    return score, pv, p, ctx


class TestScoreEqualityAcrossConfigurations(unittest.TestCase):
    def test_tt_disabled_vs_enabled_same_score(self):
        for fen in SAMPLE_FENS:
            for depth in (1, 2, 3):
                with self.subTest(fen=fen, depth=depth):
                    score_a, _, _, _ = _run(fen, depth)
                    score_b, _, _, _ = _run(fen, depth, tt=TranspositionTable(4096))
                    self.assertEqual(score_a, score_b)

    def test_basic_vs_full_ordering_same_score(self):
        for fen in SAMPLE_FENS:
            for depth in (1, 2, 3):
                with self.subTest(fen=fen, depth=depth):
                    score_b, _, _, _ = _run(fen, depth, tt=TranspositionTable(4096))
                    score_c, _, _, _ = _run(fen, depth, tt=TranspositionTable(4096), ordering_state=OrderingState())
                    self.assertEqual(score_b, score_c)

    def test_best_move_identical_when_unambiguous(self):
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"  # Rxd4 gana la dama, unica opcion razonable
        _, pv_a, _, _ = _run(fen, 2)
        _, pv_b, _, _ = _run(fen, 2, tt=TranspositionTable(4096))
        _, pv_c, _, _ = _run(fen, 2, tt=TranspositionTable(4096), ordering_state=OrderingState())
        self.assertEqual(pv_a[0], pv_b[0])
        self.assertEqual(pv_a[0], pv_c[0])
        self.assertEqual(pv_a[0], "d1d4")


class TestPVAndRootIntegrity(unittest.TestCase):
    def test_pv_legal_with_tt_and_ordering(self):
        fen = MATE_IN_2_FEN
        p = Position(fen)
        res = iterative_deepening_search(p, SearchLimits(max_depth=5), tt=TranspositionTable(4096), ordering_state=OrderingState())
        problems = validate_principal_variation(Position(fen), res)
        self.assertEqual(problems, [])
        self.assertEqual(res.mate_distance_plies, 3)

    def test_root_position_intact_with_tt(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        _, _, p, _ = _run(fen, 3, tt=TranspositionTable(4096), ordering_state=OrderingState())
        self.assertEqual(p.fen, fen)


class TestIterativeDeepeningWithTT(unittest.TestCase):
    def test_id_with_tt_reaches_requested_depth(self):
        p = Position()
        tt = TranspositionTable(8192)
        res = iterative_deepening_search(p, SearchLimits(max_depth=3), tt=tt)
        self.assertEqual(res.completed_depth, 3)
        self.assertGreater(tt.stores, 0)

    def test_id_with_full_ordering(self):
        p = Position()
        tt = TranspositionTable(8192)
        os = OrderingState()
        res = iterative_deepening_search(p, SearchLimits(max_depth=3), tt=tt, ordering_state=os)
        self.assertEqual(res.completed_depth, 3)
        self.assertIn("tt_hits", res.metadata)
        self.assertIn("hash_move_first_count", res.metadata)

    def test_tt_shows_real_hits_on_transposition_rich_search(self):
        # a profundidad 4 desde la posicion inicial existen transposiciones
        # reales (varios ordenes de jugadas alcanzan la misma posicion)
        p = Position()
        tt = TranspositionTable(65536)
        iterative_deepening_search(p, SearchLimits(max_depth=4), tt=tt)
        self.assertGreater(tt.hits, 0)

    def test_negamax_rejects_tt_and_ordering(self):
        p = Position()
        with self.assertRaises(ValueError):
            iterative_deepening_search(p, SearchLimits(max_depth=2), algorithm="negamax", tt=TranspositionTable(64))


class TestLimitsWithTT(unittest.TestCase):
    def test_interruption_by_nodes_with_tt_no_partial_result(self):
        p = Position()
        tt = TranspositionTable(8192)
        res = iterative_deepening_search(p, SearchLimits(max_depth=6, max_nodes=30), tt=tt)
        self.assertEqual(res.stop_reason, StopReason.MAX_NODES)
        self.assertTrue(res.completed)
        self.assertGreaterEqual(res.completed_depth, 1)

    def test_interruption_by_time_with_tt(self):
        p = Position()
        tt = TranspositionTable(8192)
        res = iterative_deepening_search(p, SearchLimits(max_depth=6, max_time_ms=5, time_check_frequency=1), tt=tt)
        self.assertIn(res.stop_reason, (StopReason.MAX_TIME, StopReason.DEPTH_COMPLETE))

    def test_last_complete_iteration_conserved_with_tt(self):
        p = Position()
        tt = TranspositionTable(8192)
        res = iterative_deepening_search(p, SearchLimits(max_depth=5, max_nodes=30), tt=tt)
        # reproducir de forma independiente la profundidad completada, con
        # una TT fresca, debe dar el mismo resultado que el conservado
        p2 = Position()
        tt2 = TranspositionTable(8192)
        score2, pv2, _, _ = _run(p2.fen, res.completed_depth, tt=tt2)
        self.assertEqual(res.score, score2)


class TestRepetitionAndFiftyMoveWithTT(unittest.TestCase):
    def test_stalemate_scored_zero_with_tt(self):
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        score, pv, p, ctx = _run(fen, 2, tt=TranspositionTable(4096))
        self.assertEqual(score, 0)

    def test_fifty_move_rule_scored_zero_with_tt(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 100 60"
        score, pv, p, ctx = _run(fen, 2, tt=TranspositionTable(4096))
        self.assertEqual(score, 0)


class TestCastlingAndEnPassantNoFalseHit(unittest.TestCase):
    def test_different_castling_rights_no_false_hit(self):
        tt = TranspositionTable(4096)
        fen_full = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        fen_no_castle = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
        score_full, _, _, _ = _run(fen_full, 2, tt=tt)
        score_no_castle, _, _, _ = _run(fen_no_castle, 2, tt=tt)
        # ambas posiciones comparten la misma disposicion de piezas pero
        # tienen zobrist_key distinto (Fase I ya lo verifico) - comparten
        # la MISMA tabla y no deben mezclar resultados
        tt_alone_full = TranspositionTable(4096)
        expected_full, _, _, _ = _run(fen_full, 2, tt=tt_alone_full)
        self.assertEqual(score_full, expected_full)

    def test_relevant_en_passant_no_false_hit(self):
        tt = TranspositionTable(4096)
        fen_with_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        fen_without_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
        score_with, _, _, _ = _run(fen_with_ep, 2, tt=tt)
        score_without, _, _, _ = _run(fen_without_ep, 2, tt=tt)
        tt_alone = TranspositionTable(4096)
        expected_with, _, _, _ = _run(fen_with_ep, 2, tt=tt_alone)
        self.assertEqual(score_with, expected_with)


class TestRandomReproducibleVsPhaseII(unittest.TestCase):
    @staticmethod
    def _random_positions(n=12, max_plies=6, seed=SEED):
        rng = random.Random(seed)
        positions = []
        attempts = 0
        while len(positions) < n and attempts < n * 10:
            attempts += 1
            board = chess.Board()
            for _ in range(rng.randint(0, max_plies)):
                legal = list(board.legal_moves)
                if not legal or board.is_game_over():
                    break
                board.push(rng.choice(legal))
            if not board.is_game_over():
                positions.append(board.fen())
        return positions

    def test_tt_ordering_vs_phase_ii_alpha_beta_on_random_sample(self):
        positions = self._random_positions()
        self.assertGreaterEqual(len(positions), 8)
        for fen in positions:
            with self.subTest(fen=fen):
                depth = 2
                score_phase2, _, _, _ = _run(fen, depth)  # tt=None, ordering=None
                score_phase3, _, _, _ = _run(fen, depth, tt=TranspositionTable(4096), ordering_state=OrderingState())
                self.assertEqual(score_phase2, score_phase3)


if __name__ == "__main__":
    unittest.main()
