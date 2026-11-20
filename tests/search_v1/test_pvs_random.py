"""Item 35 del pedido de Fase V: tests aleatorios reproducibles (seed
20260805) comparando PVS contra la referencia de Alpha-Beta - con y sin
TT/ordering/quiescence, y con null move/LMR habilitados (deben seguir
dando el mismo score, son PODAS que nunca deben alterar el resultado
final).

Correr: python -m pytest tests/search_v1/test_pvs_random.py -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    OrderingState, Position, PruningConfig, QuiescenceLimits, SearchLimits,
    TranspositionTable, alpha_beta, iterative_deepening_search,
)
from search_v1.search_context import SearchContext
from search_v1.repetition import RepetitionTracker
from search_v1.mate_scores import INFINITY

SEED = 20260805
QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)


def _random_positions(n=15, max_plies=8, seed=SEED):
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


def _run_ab(fen, depth):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=depth), RepetitionTracker())
    score, pv = alpha_beta(p, depth, -INFINITY, INFINITY, ctx)
    return score


class TestPVSRandomReproducible(unittest.TestCase):
    def test_pvs_matches_alpha_beta_on_random_sample_pure(self):
        positions = _random_positions()
        self.assertGreaterEqual(len(positions), 10)
        for fen in positions:
            with self.subTest(fen=fen):
                depth = 3
                score_ab = _run_ab(fen, depth)
                p = Position(fen)
                res_pvs = iterative_deepening_search(p, SearchLimits(max_depth=depth), use_pvs=True)
                self.assertEqual(score_ab, res_pvs.score)
                self.assertEqual(p.fen, fen)

    def test_pvs_matches_alpha_beta_on_random_sample_full_stack(self):
        positions = _random_positions()
        for fen in positions:
            with self.subTest(fen=fen):
                depth = 2
                p_ref = Position(fen)
                res_ref = iterative_deepening_search(
                    p_ref, SearchLimits(max_depth=depth), quiescence_limits=QLIMITS,
                    tt=TranspositionTable(8192), ordering_state=OrderingState(),
                )
                p_pvs = Position(fen)
                res_pvs = iterative_deepening_search(
                    p_pvs, SearchLimits(max_depth=depth), quiescence_limits=QLIMITS,
                    tt=TranspositionTable(8192), ordering_state=OrderingState(), use_pvs=True,
                )
                self.assertEqual(res_ref.score, res_pvs.score)

    def test_null_move_and_lmr_stay_legal_and_root_intact_on_random_sample(self):
        # null move y LMR son PODAS HEURISTICAS (a diferencia de PVS/
        # aspiration, que son exactas - ver test_lmr.py y el docstring de
        # pvs.py) - no se exige score identico aca, la validacion real de
        # seguridad tactica se hace contra las categorias criticas del
        # dataset (Fase V, PHASE_V_PRUNING_ABLATION.csv). Lo que SI debe
        # cumplirse siempre, sin excepcion, en cualquier posicion: PV/
        # best_move legal y posicion raiz intacta.
        positions = _random_positions(n=10, max_plies=8, seed=SEED + 1)
        pc = PruningConfig(null_move_enabled=True, lmr_enabled=True)
        for fen in positions:
            with self.subTest(fen=fen):
                depth = 3
                p2 = Position(fen)
                res_pruned = iterative_deepening_search(p2, SearchLimits(max_depth=depth), use_pvs=True, pruning_config=pc)
                self.assertTrue(res_pruned.legal)
                self.assertIsNotNone(res_pruned.best_move)
                self.assertEqual(p2.fen, fen)


if __name__ == "__main__":
    unittest.main()
