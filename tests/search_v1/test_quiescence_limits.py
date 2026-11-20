"""Tests de limites de seguridad de quiescence (Fase IV, items 29-33):
qdepth, qnodes, tiempo, cancelacion, ciclos de jaques.

Correr: python -m pytest tests/search_v1/test_quiescence_limits.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, QuiescenceLimits, RepetitionTracker, SearchContext,
    SearchLimits, StopReason, alpha_beta, iterative_deepening_search, quiescence,
)

TACTICAL_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


class TestQDepthLimit(unittest.TestCase):
    def test_qdepth_limit_respected_not_in_check(self):
        qlim = QuiescenceLimits(max_qdepth=1, max_check_extensions=2)
        p = Position(TACTICAL_FEN)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlim)
        score, pv = quiescence(p, -INFINITY, INFINITY, ctx, qlim, qlim.max_qdepth, qlim.max_check_extensions)
        self.assertGreaterEqual(ctx.q_max_depth_reached, 0)
        self.assertLessEqual(ctx.q_max_depth_reached, qlim.max_qdepth + qlim.max_check_extensions)

    def test_deeper_qdepth_finds_at_least_as_good_score(self):
        # mas profundidad de quiescence nunca deberia empeorar el score
        # encontrado (puede encontrar lineas mejores, nunca peores, ya
        # que la opcion de "no seguir" -stand-pat- siempre sigue disponible)
        p1, p2 = Position(TACTICAL_FEN), Position(TACTICAL_FEN)
        qlim_shallow = QuiescenceLimits(max_qdepth=1, max_check_extensions=2)
        qlim_deep = QuiescenceLimits(max_qdepth=8, max_check_extensions=4)
        ctx1 = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlim_shallow)
        ctx2 = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlim_deep)
        score1, _ = quiescence(p1, -INFINITY, INFINITY, ctx1, qlim_shallow, qlim_shallow.max_qdepth, qlim_shallow.max_check_extensions)
        score2, _ = quiescence(p2, -INFINITY, INFINITY, ctx2, qlim_deep, qlim_deep.max_qdepth, qlim_deep.max_check_extensions)
        self.assertGreaterEqual(score2, score1)


class TestQNodesLimit(unittest.TestCase):
    def test_qnodes_limit_aborts_cleanly(self):
        qlim = QuiescenceLimits(max_qdepth=8, max_qnodes=3, max_check_extensions=4)
        res = iterative_deepening_search(Position(TACTICAL_FEN), SearchLimits(max_depth=3), quiescence_limits=qlim)
        # con un techo de 3 q-nodos, alguna iteracion de profundidad debe
        # abortar por MAX_QNODES (o completar todo si la posicion es tan
        # simple que ni siquiera necesita quiescence - no es el caso aca)
        self.assertIn(res.stop_reason, (StopReason.MAX_QNODES, StopReason.DEPTH_COMPLETE))
        self.assertTrue(res.completed or res.completed_depth == 0)

    def test_qnodes_limit_never_marks_partial_as_complete(self):
        qlim = QuiescenceLimits(max_qdepth=8, max_qnodes=2, max_check_extensions=4)
        p = Position(TACTICAL_FEN)
        original_fen = p.fen
        res = iterative_deepening_search(p, SearchLimits(max_depth=4), quiescence_limits=qlim)
        self.assertEqual(p.fen, original_fen)
        if res.stop_reason == StopReason.MAX_QNODES:
            # si aborto por qnodes en la primerisima iteracion, no hay
            # iteracion completa que devolver
            self.assertFalse(res.completed_depth >= 1 and res.stop_reason != StopReason.DEPTH_COMPLETE and res.completed_depth == res.requested_depth)


class TestTimeLimit(unittest.TestCase):
    def test_time_limit_with_quiescence_no_crash(self):
        qlim = QuiescenceLimits(max_qdepth=8, max_check_extensions=4)
        res = iterative_deepening_search(
            Position(TACTICAL_FEN), SearchLimits(max_depth=6, max_time_ms=5, time_check_frequency=1),
            quiescence_limits=qlim,
        )
        self.assertIn(res.stop_reason, (StopReason.MAX_TIME, StopReason.DEPTH_COMPLETE))


class TestCancellationSafety(unittest.TestCase):
    def test_root_intact_after_qnodes_abort(self):
        qlim = QuiescenceLimits(max_qdepth=8, max_qnodes=2, max_check_extensions=4)
        p = Position(TACTICAL_FEN)
        original_fen = p.fen
        iterative_deepening_search(p, SearchLimits(max_depth=5), quiescence_limits=qlim)
        self.assertEqual(p.fen, original_fen)

    def test_last_complete_main_iteration_conserved_despite_qnode_limit(self):
        # aunque quiescence se aborte en una profundidad principal
        # posterior, la ultima iteracion PRINCIPAL completa (con su
        # propia quiescence completa) debe conservarse - nunca un
        # resultado mixto/corrupto
        qlim = QuiescenceLimits(max_qdepth=8, max_qnodes=5, max_check_extensions=4)
        res = iterative_deepening_search(Position(TACTICAL_FEN), SearchLimits(max_depth=6), quiescence_limits=qlim)
        self.assertLessEqual(res.completed_depth, 6)
        if res.completed_depth >= 1:
            self.assertTrue(res.completed)


class TestCheckCycleNoInfiniteLoop(unittest.TestCase):
    def test_repetition_detected_within_quiescence_check_sequence(self):
        # sembramos el RepetitionTracker con 2 ocurrencias previas
        # simuladas de una posicion alcanzable en 1 ply desde la raiz
        # (mismo patron ya validado en test_negamax.py::
        # test_repetition_within_search_line) - si quiescence llega a
        # esa posicion una 3ra vez, debe detectar tablas por repeticion
        # en vez de seguir buscando indefinidamente.
        from search_v1 import Move
        qlim = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)
        p = Position("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")  # en jaque, Kxe2 disponible
        after_capture = p.copy()
        after_capture.apply(Move.from_uci("e1e2"))

        rep = RepetitionTracker()
        rep.push(p)
        rep.push(after_capture)
        rep.push(after_capture)  # 2 ocurrencias previas simuladas

        ctx = SearchContext(SearchLimits(max_depth=1), rep, quiescence_limits=qlim)
        score, pv = quiescence(p, -INFINITY, INFINITY, ctx, qlim, qlim.max_qdepth, qlim.max_check_extensions)
        # no debe crashear ni colgarse (ya termino, estamos aca) - y la
        # rama e1e2 debe haberse registrado como terminal por repeticion
        self.assertGreaterEqual(ctx.q_terminal_nodes, 1)

    def test_check_extensions_exhausted_aborts_cleanly_not_corrupts(self):
        # presupuesto de extensiones de jaque deliberadamente en 0 sobre
        # una posicion en jaque con qdepth ya agotado - debe abortar de
        # forma limpia (SearchAborted), nunca devolver un valor incorrecto
        from search_v1 import SearchAborted
        qlim = QuiescenceLimits(max_qdepth=0, max_check_extensions=0)
        p = Position("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlim)
        ctx.repetition.push(p)
        with self.assertRaises(SearchAborted) as cm:
            quiescence(p, -INFINITY, INFINITY, ctx, qlim, 0, 0)
        self.assertEqual(cm.exception.reason, "max_check_extensions")
        # la posicion no queda corrupta pese al abort (apply/undo balanceados)
        self.assertEqual(p.fen, "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")


if __name__ == "__main__":
    unittest.main()
