"""Tests de integracion Transposition Table + quiescence (Fase IV, items
34-39): bounds EXACT/LOWER/UPPER, mate score almacenado/recuperado, hash
move tactico legal/ilegal.

Correr: python -m pytest tests/search_v1/test_quiescence_tt.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Bound, OrderingState, Position, QuiescenceLimits, RepetitionTracker,
    SearchContext, SearchLimits, TranspositionTable, iterative_deepening_search, quiescence,
)

QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)
TACTICAL_FEN = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"  # Rxd4 disponible


def _run_q_tt(fen, tt, alpha=-INFINITY, beta=INFINITY, qlim=QLIMITS):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlim)
    score, pv = quiescence(p, alpha, beta, ctx, qlim, qlim.max_qdepth, qlim.max_check_extensions, tt=tt)
    return score, pv, p, ctx


class TestTTBoundsInQuiescence(unittest.TestCase):
    def test_exact_bound_stored(self):
        tt = TranspositionTable(4096)
        score, pv, p, _ = _run_q_tt(TACTICAL_FEN, tt, alpha=-INFINITY, beta=INFINITY)
        entry = tt.probe_for_quiescence(p.zobrist_key, QLIMITS.max_qdepth, QLIMITS)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.EXACT)
        self.assertEqual(entry.score, score)  # ply_from_root=0 en la raiz, sin ajuste

    def test_lower_bound_stored_on_beta_cutoff(self):
        tt = TranspositionTable(4096)
        # beta muy bajo fuerza un fail-high (LOWER_BOUND) en la raiz
        score, pv, p, _ = _run_q_tt(TACTICAL_FEN, tt, alpha=-INFINITY, beta=50)
        entry = tt.probe_for_quiescence(p.zobrist_key, QLIMITS.max_qdepth, QLIMITS)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.LOWER_BOUND)

    def test_upper_bound_stored_on_fail_low(self):
        tt = TranspositionTable(4096)
        # alpha muy alto fuerza un fail-low (UPPER_BOUND) - nada supera alpha
        score, pv, p, _ = _run_q_tt(TACTICAL_FEN, tt, alpha=5000, beta=INFINITY)
        entry = tt.probe_for_quiescence(p.zobrist_key, QLIMITS.max_qdepth, QLIMITS)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.bound, Bound.UPPER_BOUND)

    def test_tt_hit_reused_on_second_call(self):
        tt = TranspositionTable(4096)
        score1, pv1, _, ctx1 = _run_q_tt(TACTICAL_FEN, tt)
        score2, pv2, _, ctx2 = _run_q_tt(TACTICAL_FEN, tt)
        self.assertEqual(score1, score2)
        self.assertEqual(pv1, pv2)
        self.assertGreater(ctx2.q_tt_hits, 0)


class TestMateScoreInQuiescenceTT(unittest.TestCase):
    def test_mate_score_stored_and_recovered(self):
        tt = TranspositionTable(4096)
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        score1, pv1, p, _ = _run_q_tt(fen, tt)
        from search_v1 import is_mate_score
        self.assertTrue(is_mate_score(score1))
        score2, pv2, _, ctx2 = _run_q_tt(fen, tt)
        self.assertEqual(score1, score2)
        self.assertEqual(pv1, pv2)
        self.assertGreater(ctx2.q_tt_hits, 0)


class TestTacticalHashMove(unittest.TestCase):
    def test_legal_hash_move_used_first(self):
        p = Position(TACTICAL_FEN)
        from search_v1 import generate_tactical_moves, order_tactical_moves
        moves = generate_tactical_moves(p)
        ordered = order_tactical_moves(p, moves, hash_move_uci="d1d4")
        self.assertEqual(ordered[0].uci, "d1d4")

    def test_illegal_hash_move_ignored(self):
        p = Position(TACTICAL_FEN)
        from search_v1 import generate_tactical_moves, order_tactical_moves
        moves = generate_tactical_moves(p)
        ordered = order_tactical_moves(p, moves, hash_move_uci="a1a8")  # no es un movimiento tactico legal aca
        self.assertNotIn("a1a8", [lm.uci for lm in ordered])
        self.assertEqual(ordered[0].uci, "d1d4")  # sigue siendo la mejor jugada tactica real

    def test_hash_move_from_tt_actually_used_in_quiescence(self):
        tt = TranspositionTable(4096)
        # primera corrida llena la TT con d1d4 como best_move (guardada
        # con qdepth_remaining=QLIMITS.max_qdepth=6)
        _run_q_tt(TACTICAL_FEN, tt)
        # se sondea con qdepth_remaining=5 (no 0): debe quedar en el MISMO
        # regimen de allow_checks que el guardado (ver BUG REAL #2 en
        # transposition_table.py) para que la reutilizacion sea valida -
        # sondear con 0 cruzaria el limite de tactical_checks_max_qdepth y
        # la entrada se rechazaria correctamente (insuficiente por
        # regimen, no por profundidad).
        entry = tt.probe_for_quiescence(Position(TACTICAL_FEN).zobrist_key, 5, QLIMITS)
        self.assertEqual(entry.best_move, "d1d4")


class TestQuiescenceTTRegressionRealDataset(unittest.TestCase):
    """Regresion permanente para los 2 bugs reales de TT+quiescence
    encontrados en la validacion de Fase IV contra el dataset real (ver
    docstring de transposition_table.py, "BUG REAL #1" y "BUG REAL #2").
    Ambos se manifestaban en la MISMA posicion (position_id
    d5c599830621cd23), especificamente en la rama tras 1.Rd1 (b1d1):
    oracle (sin TT) daba -530, la version con TT daba -630 - primero por
    colision de codificacion de profundidad durante extensiones de jaque
    (bug #1), y luego, ya con el bug #1 corregido, por reutilizacion de
    una entrada calculada con jaques tacticos permitidos para una
    consulta en el regimen sin jaques (bug #2)."""

    FEN_D5C599830621CD23 = "6k1/7p/pQ4p1/1p1qp3/1P2b3/P7/2r5/1R2K3 w - - 2 39"

    def test_oracle_matches_optimized_score(self):
        qlimits = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)

        p_oracle = Position(self.FEN_D5C599830621CD23)
        res_oracle = iterative_deepening_search(
            p_oracle, SearchLimits(max_depth=2), quiescence_limits=qlimits, tt=None, ordering_state=None,
        )

        p_opt = Position(self.FEN_D5C599830621CD23)
        res_opt = iterative_deepening_search(
            p_opt, SearchLimits(max_depth=2), quiescence_limits=qlimits,
            tt=TranspositionTable(8192), ordering_state=OrderingState(),
        )

        self.assertEqual(res_oracle.score, -530)
        self.assertEqual(res_oracle.score, res_opt.score)
        self.assertEqual(res_oracle.best_move, res_opt.best_move)


if __name__ == "__main__":
    unittest.main()
