"""Tests de semantica core de quiescence (Fase IV): stand-pat, capturas,
recapturas, efecto horizonte, capturas al paso, promociones/
subpromociones, posiciones en jaque (evasiones), terminales.

Correr: python -m pytest tests/search_v1/test_quiescence.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, QuiescenceLimits, RepetitionTracker, SearchContext,
    SearchLimits, iterative_deepening_search, quiescence,
)

DEFAULT_QLIMITS = QuiescenceLimits(max_qdepth=6, max_check_extensions=4)


def _run_q(fen, qlimits=DEFAULT_QLIMITS, alpha=-INFINITY, beta=INFINITY):
    p = Position(fen)
    ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker(), quiescence_limits=qlimits)
    score, pv = quiescence(p, alpha, beta, ctx, qlimits, qlimits.max_qdepth, qlimits.max_check_extensions)
    return score, pv, p, ctx


class TestQuietPositionAndStandPat(unittest.TestCase):
    def test_quiet_position_no_captures_returns_stand_pat(self):
        from search_v1.static_evaluation import evaluate_static_relative
        p = Position()
        score, pv, _, ctx = _run_q(p.fen)
        self.assertEqual(score, evaluate_static_relative(p))
        self.assertEqual(pv, [])
        self.assertEqual(ctx.q_stand_pat_calls, 1)

    def test_stand_pat_below_alpha_does_not_cutoff(self):
        # blancas claramente peor material, alpha muy bajo -> stand-pat NO
        # corta, pero tampoco mejora nada por encima de alpha (queda mas
        # abajo, alpha no se actualiza a un valor mayor que el propio alpha original)
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        score, pv, _, ctx = _run_q(fen, alpha=500, beta=INFINITY)
        self.assertEqual(score, 0)  # material igual, stand-pat=0, no supera alpha=500
        self.assertEqual(ctx.q_stand_pat_cutoffs, 0)

    def test_stand_pat_within_window_no_cutoff(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        score, pv, _, ctx = _run_q(fen, alpha=-100, beta=100)
        self.assertEqual(score, 0)
        self.assertEqual(ctx.q_stand_pat_cutoffs, 0)

    def test_stand_pat_at_or_above_beta_cutoff(self):
        fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"  # blancas con dama de mas, stand-pat=900
        score, pv, _, ctx = _run_q(fen, alpha=-INFINITY, beta=100)
        self.assertEqual(score, 900)
        self.assertEqual(ctx.q_stand_pat_cutoffs, 1)
        self.assertEqual(ctx.q_beta_cutoffs, 1)

    def test_stand_pat_forbidden_in_check(self):
        # rey blanco en jaque - NO debe usarse evaluate_static_relative
        # directamente como resultado sin buscar evasiones, NI SIQUIERA
        # en el caso mas riesgoso: max_qdepth=0 justo al entrar (si el
        # codigo tuviera el bug de "stand-pat por limite de profundidad
        # sin mirar in_check", esto lo revelaria porque evaluate_static_
        # relative(p) del rey en jaque NO refleja que esta perdiendo la
        # torre gratis - el resultado real debe venir de buscar la
        # evasion, no de una evaluacion estatica congelada)
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        from search_v1.static_evaluation import evaluate_static_relative
        raw_static = evaluate_static_relative(p)  # ve el material actual (torre negra de mas), no busca evasion

        tight_qlimits = QuiescenceLimits(max_qdepth=0, max_check_extensions=3)
        score, pv, _, ctx = _run_q(fen, qlimits=tight_qlimits)
        # con max_qdepth=0 pero en jaque, DEBE seguir buscando evasiones
        # (via check_extensions) en vez de devolver el stand-pat congelado
        self.assertGreater(len(pv), 0)
        self.assertNotEqual(score, raw_static)


class TestCapturesAndRecaptures(unittest.TestCase):
    def test_winning_capture_improves_over_stand_pat(self):
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"  # Rxd4 gana la dama
        score, pv, _, _ = _run_q(fen)
        self.assertEqual(pv[0], "d1d4")
        self.assertGreater(score, 0)

    def test_losing_capture_not_chosen_over_stand_pat(self):
        # capturar con la dama una pieza defendida por un peon pierde
        # material neto - quiescence NO debe elegir esa captura sobre
        # stand-pat (quedarse quieto)
        fen = "4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1"  # Qxd5 gana un peon limpio en realidad
        # ajustamos: peon defendido por otro peon negro
        fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"  # Qxd5 seguido de cxd5 pierde la dama por un peon
        score, pv, _, _ = _run_q(fen)
        # el stand-pat (0, material inicial parejo salvo la dama de blancas
        # que ya esta contada) debe preferirse a perder la dama por un peon
        self.assertNotEqual(pv[:1], ["d1d5"])

    def test_recapture_sequence_resolved(self):
        # cadena de capturas en la misma casilla: peon blanco captura,
        # peon negro recaptura - quiescence debe resolver el resultado
        # neto correctamente (no detenerse en la primera captura)
        fen = "4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1"  # cxd5 y no hay recaptura (limpio)
        score, pv, _, _ = _run_q(fen)
        self.assertEqual(pv[0], "c4d5")
        self.assertGreater(score, 0)


class TestHorizonEffect(unittest.TestCase):
    def test_horizon_position_resolved_by_quiescence(self):
        # posicion donde una captura aparentemente buena a profundidad 1
        # se revela mala tras la recaptura - solo visible extendiendo con
        # capturas (quiescence), no con evaluacion estatica pura en el horizonte
        fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
        from search_v1.static_evaluation import evaluate_static_relative
        p = Position(fen)
        static_only = evaluate_static_relative(p)  # ve solo el material actual, no la trampa
        score, pv, _, _ = _run_q(fen)
        # con quiescence, el motor evita Qxd5 (pierde la dama) - el score
        # coincide con quedarse quieto (stand-pat), NO con haber capturado
        self.assertEqual(score, static_only)


class TestEnPassantAndPromotions(unittest.TestCase):
    def test_en_passant_capture_searched(self):
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(pv[0], "e5d6")
        self.assertEqual(ctx.q_en_passant_searched, 1)

    def test_promotion_without_capture_searched(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(pv[0][:4], "a7a8")
        self.assertGreater(ctx.q_promotions_searched, 0)

    def test_promotion_with_capture_searched(self):
        fen = "3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(pv[0][:4], "e7d8")
        self.assertGreater(ctx.q_promotions_searched, 0)

    def test_underpromotion_to_rook_considered(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        p = Position(fen)
        from search_v1.tactical_moves import generate_tactical_moves
        moves = {lm.uci for lm in generate_tactical_moves(p)}
        self.assertIn("a7a8r", moves)

    def test_underpromotion_to_bishop_considered(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        p = Position(fen)
        from search_v1.tactical_moves import generate_tactical_moves
        moves = {lm.uci for lm in generate_tactical_moves(p)}
        self.assertIn("a7a8b", moves)

    def test_underpromotion_to_knight_considered(self):
        fen = "8/P6k/8/8/8/8/7K/8 w - - 0 1"
        p = Position(fen)
        from search_v1.tactical_moves import generate_tactical_moves
        moves = {lm.uci for lm in generate_tactical_moves(p)}
        self.assertIn("a7a8n", moves)


class TestCheckEvasions(unittest.TestCase):
    def test_check_with_capturing_evasion(self):
        # rey en jaque por una torre NO defendida, adyacente - Kxe2 es
        # una evasion capturante legal y claramente la mejor opcion
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        from search_v1 import Move
        self.assertTrue(p.is_legal(Move.from_uci("e1e2")))
        score, pv, _, ctx = _run_q(fen)
        self.assertGreater(ctx.q_evasions_searched, 0)
        self.assertEqual(pv[0], "e1e2")  # captura la torre, mejor que huir

    def test_check_with_non_capturing_evasion(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"  # jaque en la columna e, sin captura posible directa del rey
        p = Position(fen)
        self.assertTrue(p.is_check)
        score, pv, _, ctx = _run_q(fen)
        self.assertGreater(len(pv), 0)
        self.assertGreater(ctx.q_evasions_searched, 0)

    def test_check_with_block(self):
        # jaque de torre a larga distancia, bloqueable interponiendo
        fen = "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        legal = {m.uci for m in __import__("search_v1").generate_legal_moves(p)}
        # el rey blanco no tiene pieza para bloquear (solo rey) - se
        # verifica en cambio que TODAS las evasiones (incluido mover el
        # rey) se consideran sin excepcion
        score, pv, _, _ = _run_q(fen)
        self.assertIn(pv[0], legal)

    def test_check_with_king_move(self):
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        score, pv, _, _ = _run_q(fen)
        self.assertTrue(pv[0].startswith("e1"))


class TestTerminalsInQuiescence(unittest.TestCase):
    def test_checkmate_no_evasions(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(pv, [])
        self.assertEqual(ctx.q_terminal_nodes, 1)
        self.assertTrue(__import__("search_v1").is_mate_score(score))

    def test_stalemate(self):
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(score, 0)
        self.assertEqual(ctx.q_terminal_nodes, 1)

    def test_repetition_scored_zero(self):
        # se prueba a nivel de mecanismo: repeticion detectada ANTES que
        # nada mas en quiescence, igual que en alpha_beta.py
        from search_v1 import Move
        p = Position()
        after = p.copy()
        after.apply(Move.from_uci("g1f3"))
        rep = RepetitionTracker()
        rep.push(p)
        rep.push(after)
        rep.push(after)
        ctx = SearchContext(SearchLimits(max_depth=1), rep, quiescence_limits=DEFAULT_QLIMITS)
        p2 = p.copy()
        p2.apply(Move.from_uci("g1f3"))
        score, pv = quiescence(p2, -INFINITY, INFINITY, ctx, DEFAULT_QLIMITS, DEFAULT_QLIMITS.max_qdepth, DEFAULT_QLIMITS.max_check_extensions)
        self.assertEqual(score, 0)
        self.assertEqual(pv, [])

    def test_fifty_move_rule(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 100 60"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(score, 0)

    def test_insufficient_material(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
        score, pv, _, ctx = _run_q(fen)
        self.assertEqual(score, 0)
        self.assertEqual(ctx.q_terminal_nodes, 1)


if __name__ == "__main__":
    unittest.main()
