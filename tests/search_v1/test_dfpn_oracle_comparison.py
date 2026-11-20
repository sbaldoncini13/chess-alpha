"""Tests de reproducibilidad y comparacion contra el oraculo exhaustivo de
referencia (Fase VI, items 43/46/47): prueba reproducible, comparacion
DFPN vs busqueda exhaustiva independiente, tests aleatorios reproducibles
en arboles chicos (seed 20260805).

Correr: python -m pytest tests/search_v1/test_dfpn_oracle_comparison.py -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import DFPNLimits, Position, ProofStatus, find_forced_mate, verify_mate_proof
from search_v1.proof_verifier import exhaustive_mate_oracle

SEED = 20260805

_SMALL_FIXTURES = [
    "4k3/8/4K3/8/8/8/8/7Q w - - 0 1",   # mate en 1
    "k7/8/2K5/8/8/8/8/7Q w - - 0 1",     # mate en 2
    "6k1/8/6K1/8/8/7R/8/8 w - - 0 1",    # mate en 3
    "k7/2P5/K7/8/8/8/8/8 w - - 0 1",     # mate en 1 via promocion
    "4k3/8/4K3/8/8/8/8/7Q b - - 0 1",    # sin mate (lado equivocado)
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",     # sin mate (material insuficiente)
]


class TestReproducibleProof(unittest.TestCase):
    def test_same_call_reproducible(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        results = []
        for _ in range(3):
            p = Position(fen)
            res = find_forced_mate(p, DFPNLimits(max_nodes=5000))
            results.append((res.proof_status, res.mate_distance_plies, tuple(res.principal_variation), res.nodes))
        self.assertEqual(len(set(results)), 1)


class TestDFPNMatchesExhaustiveOracle(unittest.TestCase):
    def test_all_small_fixtures_agree_with_oracle(self):
        for fen in _SMALL_FIXTURES:
            with self.subTest(fen=fen):
                p_dfpn = Position(fen)
                dfpn_res = find_forced_mate(p_dfpn, DFPNLimits(max_nodes=20000, max_depth_plies=None))

                p_oracle = Position(fen)
                oracle_distance, oracle_pv = exhaustive_mate_oracle(p_oracle, max_depth_plies=6)

                dfpn_found_mate = dfpn_res.proof_status == ProofStatus.PROVEN_MATE
                oracle_found_mate = oracle_distance is not None

                self.assertEqual(dfpn_found_mate, oracle_found_mate, f"discrepancia en {fen}")
                if dfpn_found_mate:
                    self.assertEqual(dfpn_res.mate_distance_plies, oracle_distance)
                    # 0 falsos mates: la PV de DFPN debe verificarse independientemente
                    self.assertEqual(verify_mate_proof(Position(fen), dfpn_res), [])

    def test_zero_false_mates_and_zero_false_no_mates(self):
        n_false_mate = 0
        n_false_no_mate = 0
        for fen in _SMALL_FIXTURES:
            p_dfpn = Position(fen)
            dfpn_res = find_forced_mate(p_dfpn, DFPNLimits(max_nodes=20000, max_depth_plies=None))
            p_oracle = Position(fen)
            oracle_distance, _ = exhaustive_mate_oracle(p_oracle, max_depth_plies=6)

            if dfpn_res.proof_status == ProofStatus.PROVEN_MATE and oracle_distance is None:
                n_false_mate += 1
            if dfpn_res.proof_status == ProofStatus.PROVEN_NO_MATE and oracle_distance is not None:
                n_false_no_mate += 1
        self.assertEqual(n_false_mate, 0)
        self.assertEqual(n_false_no_mate, 0)


class TestRandomReproducibleSmallTrees(unittest.TestCase):
    @staticmethod
    def _random_small_endgame_positions(n=8, seed=SEED):
        # genera posiciones chicas (K+Q vs K) reproducibles - suficiente
        # material para tener mates reales sin arboles demasiado grandes.
        rng = random.Random(seed)
        positions = []
        attempts = 0
        while len(positions) < n and attempts < n * 30:
            attempts += 1
            squares = rng.sample(range(64), 3)
            board = chess.Board(None)
            board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(squares[2], chess.Piece(chess.QUEEN, chess.WHITE))
            board.turn = chess.WHITE
            if board.status() != chess.STATUS_VALID:
                continue
            positions.append(board.fen())
        return positions

    def test_random_small_endgames_reproducible_and_consistent(self):
        positions = self._random_small_endgame_positions()
        self.assertGreaterEqual(len(positions), 5)
        for fen in positions:
            with self.subTest(fen=fen):
                results = []
                last_res = None
                for _ in range(2):
                    p = Position(fen)
                    res = find_forced_mate(p, DFPNLimits(max_nodes=8000, max_depth_plies=10))
                    results.append((res.proof_status, res.mate_distance_plies, tuple(res.principal_variation)))
                    last_res = res
                self.assertEqual(results[0], results[1])  # reproducible
                if last_res.proof_status == ProofStatus.PROVEN_MATE:
                    problems = verify_mate_proof(Position(fen), last_res)
                    self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
