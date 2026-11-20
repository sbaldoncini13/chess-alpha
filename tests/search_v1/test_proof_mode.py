"""Tests de proof_mode (Fase VI, items 35-38): proof_mode desactiva
incondicionalmente null move/LMR/futility/razoring dentro de PVS, sin
importar lo que diga `pruning_config` - "una prueba exacta no puede
depender de una poda heuristica sin verificacion".

Correr: python -m pytest tests/search_v1/test_proof_mode.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, PruningConfig, RepetitionTracker, SearchContext,
    SearchLimits, TranspositionTable, OrderingState, iterative_deepening_search, pvs,
)

_TACTICAL_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


def _run_with_all_pruning(fen, depth, proof_mode):
    pc = PruningConfig(
        null_move_enabled=True, lmr_enabled=True, futility_enabled=True, razoring_enabled=True,
    )
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=depth), RepetitionTracker())
    score, pv = pvs(
        p, depth, -INFINITY, INFINITY, ctx,
        tt=TranspositionTable(16384), ordering_state=OrderingState(),
        pruning_config=pc, proof_mode=proof_mode,
    )
    return score, pv, ctx


class TestProofModeDisablesNullMove(unittest.TestCase):
    def test_null_move_never_attempted_in_proof_mode(self):
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=True)
        self.assertEqual(ctx.null_move_attempts, 0)
        self.assertEqual(ctx.null_move_cutoffs, 0)

    def test_null_move_does_attempt_when_proof_mode_false(self):
        # control positivo - confirma que la MISMA config, sin proof_mode,
        # SI intenta null move (la guarda de arriba es especifica de
        # proof_mode, no un bug que desactiva null move siempre).
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=False)
        self.assertGreater(ctx.null_move_attempts, 0)


class TestProofModeDisablesLMR(unittest.TestCase):
    def test_lmr_never_reduces_in_proof_mode(self):
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=True)
        self.assertEqual(ctx.lmr_reductions_applied, 0)
        self.assertEqual(ctx.lmr_researches, 0)

    def test_lmr_does_reduce_when_proof_mode_false(self):
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=False)
        self.assertGreater(ctx.lmr_reductions_applied, 0)


class TestProofModeDisablesFutility(unittest.TestCase):
    def test_futility_never_prunes_in_proof_mode(self):
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=True)
        self.assertEqual(ctx.futility_prunes, 0)


class TestProofModeDisablesRazoring(unittest.TestCase):
    def test_razoring_never_prunes_in_proof_mode(self):
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 5, proof_mode=True)
        self.assertEqual(ctx.razoring_prunes, 0)


class TestProofModeStillAllowsOrdering(unittest.TestCase):
    def test_hash_move_and_ordering_still_used_in_proof_mode(self):
        # el orden de jugadas (hash move/killers/history/countermove)
        # sigue permitido en proof_mode - solo cambia el COSTO, nunca el
        # conjunto de jugadas legales exploradas.
        _, _, ctx = _run_with_all_pruning(_TACTICAL_FEN, 4, proof_mode=True)
        self.assertGreater(ctx.tt_cutoffs + ctx.alpha_beta_cutoffs, 0)

    def test_proof_mode_result_matches_no_pruning_result(self):
        # proof_mode=True con pruning_config lleno de podas activas debe
        # dar el MISMO resultado que pruning_config=None (ninguna poda) -
        # confirma que proof_mode realmente anula pruning_config, no solo
        # reduce su efecto.
        pc = PruningConfig(
            null_move_enabled=True, lmr_enabled=True, futility_enabled=True, razoring_enabled=True,
        )
        p1 = Position(_TACTICAL_FEN)
        res1 = iterative_deepening_search(
            p1, SearchLimits(max_depth=4), tt=TranspositionTable(16384), ordering_state=OrderingState(),
            use_pvs=True, pruning_config=None,
        )
        p2 = Position(_TACTICAL_FEN)
        res2 = iterative_deepening_search(
            p2, SearchLimits(max_depth=4), tt=TranspositionTable(16384), ordering_state=OrderingState(),
            use_pvs=True, pruning_config=pc, proof_mode=True,
        )
        self.assertEqual(res1.score, res2.score)


if __name__ == "__main__":
    unittest.main()
