"""Tests de Alpha-Beta contra Negamax (Fase II): igualdad de score,
reduccion de nodos, cutoffs, ordenamiento de movimientos, comparacion
aleatoria reproducible (seed=20260805).

Correr: python -m pytest tests/search_v1/test_alpha_beta.py -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    INFINITY, Position, RepetitionTracker, SearchContext, SearchLimits,
    alpha_beta, negamax,
)

SEED = 20260805
MATE_IN_1_FEN = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
MATE_IN_2_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"


def _ctx(max_depth=5):
    return SearchContext(SearchLimits(max_depth=max_depth), RepetitionTracker())


class TestAlphaBetaVsNegamaxScores(unittest.TestCase):
    SAMPLE_FENS = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 4",
        MATE_IN_1_FEN,
        MATE_IN_2_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    ]

    def test_identical_scores_at_shallow_depth(self):
        # depth<=2 para todas las posiciones (incluye Kiwipete, 48 jugadas
        # legales - profundidad 3 sin poda ahi ya es ~110k nodos, se
        # cubre por separado solo en posiciones simples via
        # test_mate_in_2_agrees, para no competir de mas por CPU)
        for fen in self.SAMPLE_FENS:
            for depth in (1, 2):
                with self.subTest(fen=fen, depth=depth):
                    p_neg, p_ab = Position(fen), Position(fen)
                    score_neg, _ = negamax(p_neg, depth, _ctx())
                    score_ab, _ = alpha_beta(p_ab, depth, -INFINITY, INFINITY, _ctx())
                    self.assertEqual(score_neg, score_ab)

    def test_best_move_identical_when_unambiguous(self):
        # posicion con una unica jugada claramente mejor (captura de dama
        # colgada) - no deberia haber empates
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"  # Rxd4 gana la dama
        p_neg, p_ab = Position(fen), Position(fen)
        _, pv_neg = negamax(p_neg, 2, _ctx())
        _, pv_ab = alpha_beta(p_ab, 2, -INFINITY, INFINITY, _ctx())
        self.assertEqual(pv_neg[0], pv_ab[0])
        self.assertEqual(pv_neg[0], "d1d4")

    def test_mate_in_1_agrees(self):
        p_neg, p_ab = Position(MATE_IN_1_FEN), Position(MATE_IN_1_FEN)
        score_neg, pv_neg = negamax(p_neg, 2, _ctx())
        score_ab, pv_ab = alpha_beta(p_ab, 2, -INFINITY, INFINITY, _ctx())
        self.assertEqual(score_neg, score_ab)
        self.assertEqual(pv_neg, pv_ab)  # h1a8 es primero en orden UCI en ambos esquemas

    def test_mate_in_2_agrees(self):
        p_neg, p_ab = Position(MATE_IN_2_FEN), Position(MATE_IN_2_FEN)
        score_neg, pv_neg = negamax(p_neg, 3, _ctx())
        score_ab, pv_ab = alpha_beta(p_ab, 3, -INFINITY, INFINITY, _ctx())
        self.assertEqual(score_neg, score_ab)
        self.assertEqual(pv_neg, pv_ab)


class TestAlphaBetaNodeReduction(unittest.TestCase):
    def test_fewer_or_equal_nodes_than_negamax(self):
        for fen in TestAlphaBetaVsNegamaxScores.SAMPLE_FENS:
            for depth in (1, 2):
                with self.subTest(fen=fen, depth=depth):
                    p_neg, p_ab = Position(fen), Position(fen)
                    ctx_neg, ctx_ab = _ctx(), _ctx()
                    negamax(p_neg, depth, ctx_neg)
                    alpha_beta(p_ab, depth, -INFINITY, INFINITY, ctx_ab)
                    self.assertLessEqual(ctx_ab.nodes, ctx_neg.nodes)

    def test_cutoffs_recorded_on_tactical_position(self):
        p = Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        ctx = _ctx()
        alpha_beta(p, 3, -INFINITY, INFINITY, ctx)
        self.assertGreater(ctx.alpha_beta_cutoffs, 0)

    def test_negamax_never_cuts_off(self):
        p = Position()
        ctx = _ctx()
        negamax(p, 2, ctx)
        self.assertEqual(ctx.alpha_beta_cutoffs, 0)


class TestAlphaBetaMoveOrdering(unittest.TestCase):
    def test_mate_move_ordered_before_capture(self):
        from search_v1.alpha_beta import _order_moves
        from search_v1.move_generator import generate_legal_moves
        p = Position(MATE_IN_1_FEN)
        moves = generate_legal_moves(p)
        ordered = _order_moves(p, moves)
        # varias jugadas de dama entregan mate inmediato en esta posicion
        # (a8/b8/c8/g8/h8, todas por dominio de la fila 8 con apoyo del
        # rey blanco) - el desempate UCI determinista hace que "h1a8" sea
        # la primera del tier 0 (mate), antes que cualquier movimiento de
        # rey (que ordena antes alfabeticamente por casilla de origen
        # "e6..." pero no pertenece al tier de mate)
        self.assertEqual(ordered[0].uci, "h1a8")

    def test_capture_ordered_before_quiet_move(self):
        from search_v1.alpha_beta import _order_moves
        from search_v1.move_generator import generate_legal_moves
        fen = "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        ordered = _order_moves(p, moves)
        self.assertEqual(ordered[0].uci, "d1d4")  # unica captura disponible


class TestAlphaBetaRandomReproducible(unittest.TestCase):
    """Item 27: tests aleatorios reproducibles contra Negamax de
    referencia. Muestra de posiciones legales generada mediante una
    caminata aleatoria determinista (seed=20260805) desde la posicion
    inicial - no requiere el dataset, es autocontenida y liviana."""

    @staticmethod
    def _random_positions(n=15, max_plies=6, seed=SEED):
        rng = random.Random(seed)
        positions = []
        attempts = 0
        while len(positions) < n and attempts < n * 10:
            attempts += 1
            board = chess.Board()
            n_plies = rng.randint(0, max_plies)
            for _ in range(n_plies):
                legal = list(board.legal_moves)
                if not legal or board.is_game_over():
                    break
                board.push(rng.choice(legal))
            if not board.is_game_over():
                positions.append(board.fen())
        return positions

    def test_negamax_vs_alpha_beta_on_random_sample(self):
        positions = self._random_positions()
        self.assertGreaterEqual(len(positions), 10)
        for fen in positions:
            for depth in (1, 2):
                with self.subTest(fen=fen, depth=depth):
                    p_neg, p_ab = Position(fen), Position(fen)
                    score_neg, _ = negamax(p_neg, depth, _ctx())
                    score_ab, _ = alpha_beta(p_ab, depth, -INFINITY, INFINITY, _ctx())
                    self.assertEqual(score_neg, score_ab)
                    self.assertEqual(p_neg.fen, fen)
                    self.assertEqual(p_ab.fen, fen)


if __name__ == "__main__":
    unittest.main()
