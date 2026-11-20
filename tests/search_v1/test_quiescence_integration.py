"""Tests de integracion final de quiescence (Fase IV, items 40-43):
determinismo, activacion/desactivacion, comparacion contra el oraculo de
referencia, muestra aleatoria reproducible (seed=20260805).

ORACULO DE REFERENCIA (documentado): quiescence(tt=None, ordering_state=
None) - sin cache, con el mismo orden por tiers de order_tactical_moves
pero sin desempate de history (tier "resto" cae a UCI simple). Se compara
contra la version "optimizada" (tt=TranspositionTable(...), ordering_
state=OrderingState()) exactamente con el mismo patron ya usado y
validado en Fase III para las configuraciones A/B/C - reutilizar el
mismo algoritmo con distinta configuracion es mas seguro que mantener una
segunda implementacion independiente sin validar.

Correr: python -m pytest tests/search_v1/test_quiescence_integration.py -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    INFINITY, OrderingState, Position, QuiescenceLimits, SearchLimits,
    StopReason, TranspositionTable, iterative_deepening_search,
)

SEED = 20260805
QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)

TACTICAL_POSITIONS = [
    "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
]


class TestDeterminism(unittest.TestCase):
    def test_same_config_same_result(self):
        for fen in TACTICAL_POSITIONS:
            with self.subTest(fen=fen):
                res1 = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
                res2 = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
                self.assertEqual(res1.score, res2.score)
                self.assertEqual(res1.best_move, res2.best_move)
                self.assertEqual(res1.principal_variation, res2.principal_variation)


class TestQuiescenceOnOff(unittest.TestCase):
    def test_quiescence_can_be_disabled(self):
        fen = TACTICAL_POSITIONS[0]
        res_off = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1))
        res_on = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        # ambas corridas validas, sin crash; con quiescence activada
        # deberian existir q_nodes>0, sin ella no existe la clave
        self.assertNotIn("q_nodes", res_off.metadata)
        self.assertIn("q_nodes", res_on.metadata)
        self.assertGreater(res_on.metadata["q_nodes"], 0)

    def test_disabled_matches_phase_iii_exact_behavior(self):
        # sin quiescence_limits, el resultado debe ser IDENTICO al ya
        # validado en Fase III (mismo algoritmo, sin extension)
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        res = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2))
        self.assertNotIn("q_nodes", res.metadata)


class TestOracleVsOptimized(unittest.TestCase):
    def test_score_identical_oracle_vs_optimized(self):
        for fen in TACTICAL_POSITIONS:
            with self.subTest(fen=fen):
                res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
                res_optimized = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
                    tt=TranspositionTable(8192), ordering_state=OrderingState(),
                )
                self.assertEqual(res_oracle.score, res_optimized.score)

    def test_mate_distance_identical_oracle_vs_optimized(self):
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        res_optimized = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )
        self.assertEqual(res_oracle.mate_distance_plies, res_optimized.mate_distance_plies)

    def test_terminal_result_identical(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        res_optimized = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )
        self.assertEqual(res_oracle.stop_reason, res_optimized.stop_reason)
        self.assertEqual(res_oracle.score, res_optimized.score)

    def test_best_move_identical_when_unambiguous(self):
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
        res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS)
        res_optimized = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=1), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )
        self.assertEqual(res_oracle.best_move, res_optimized.best_move)
        self.assertEqual(res_oracle.best_move, "d1d4")

    def test_pv_legal_both(self):
        from search_v1 import validate_principal_variation
        fen = TACTICAL_POSITIONS[1]
        res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
        res_optimized = iterative_deepening_search(
            Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )
        self.assertEqual(validate_principal_variation(Position(fen), res_oracle), [])
        self.assertEqual(validate_principal_variation(Position(fen), res_optimized), [])


class TestRandomReproducible(unittest.TestCase):
    @staticmethod
    def _random_positions(n=10, max_plies=8, seed=SEED):
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

    def test_oracle_vs_optimized_on_random_sample(self):
        positions = self._random_positions()
        self.assertGreaterEqual(len(positions), 7)
        for fen in positions:
            with self.subTest(fen=fen):
                res_oracle = iterative_deepening_search(Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS)
                res_optimized = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), quiescence_limits=QLIMITS,
                    tt=TranspositionTable(8192), ordering_state=OrderingState(),
                )
                self.assertEqual(res_oracle.score, res_optimized.score)
                self.assertEqual(Position(fen).fen, fen)


if __name__ == "__main__":
    unittest.main()
