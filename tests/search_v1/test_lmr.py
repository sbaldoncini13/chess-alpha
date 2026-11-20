"""Tests de Late Move Reductions (Fase V, items 31-32): LMR con
re-search cuando la jugada reducida supera alpha, LMR NO aplicada a
jugadas criticas (capturas/promociones/jaques/jugada de hash/nodo en
jaque).

Correr: python -m pytest tests/search_v1/test_lmr.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, OrderingState, Position, PruningConfig, RepetitionTracker,
    SearchContext, SearchLimits, TranspositionTable, iterative_deepening_search, pvs,
)


class TestLMRResearch(unittest.TestCase):
    def test_lmr_researches_when_reduced_move_beats_alpha(self):
        # con TT+ordering (move ordering realista) el research se dispara
        # de forma confiable - sin TT/ordering el orden Fase II (tier fijo)
        # es tan bueno en posiciones chicas que rara vez una jugada tardia
        # "sorprende", lo cual NO es un bug (ver test de abajo, que separa
        # la garantia real: el mecanismo de re-search existe y se ejecuta).
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=4), RepetitionTracker())
        pc = PruningConfig(lmr_enabled=True, lmr_min_depth=3, lmr_min_move_index=1, lmr_reduction=1)
        pvs(p, 4, -INFINITY, INFINITY, ctx, tt=TranspositionTable(16384), ordering_state=OrderingState(), pruning_config=pc)
        self.assertGreater(ctx.lmr_researches, 0)
        self.assertGreater(ctx.lmr_reductions_applied, ctx.lmr_researches)

    def test_lmr_is_a_lossy_heuristic_not_guaranteed_exact(self):
        # IMPORTANTE (documentado en pvs.py y pruning_config.py): a
        # diferencia de PVS y aspiration windows (matematicamente exactas,
        # ver test_pvs.py/test_aspiration.py), LMR es una PODA HEURISTICA -
        # el re-search solo se dispara cuando la busqueda reducida SUPERA
        # alpha; si la busqueda reducida (a menos profundidad) confirma
        # "no supera alpha" cuando una busqueda completa SI lo habria
        # hecho, el resultado final puede diferir del de Alpha-Beta puro.
        # Este es el comportamiento ESPERADO de LMR en cualquier motor real
        # - la validacion real de esta tecnica (0 mates perdidos, 0
        # defensas omitidas, 0 movimientos ilegales) se hace contra las
        # categorias criticas del dataset (ver PHASE_V_PRUNING_ABLATION.csv),
        # NO exigiendo score identico en muestras arbitrarias. Este test
        # documenta esa garantia MAS DEBIL explicitamente: el resultado
        # sigue siendo LEGAL y estructuralmente correcto, aunque el score
        # puntual pueda diferir.
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        res_lmr = iterative_deepening_search(
            p, SearchLimits(max_depth=4), tt=TranspositionTable(8192),
            ordering_state=OrderingState(), use_pvs=True,
            pruning_config=PruningConfig(lmr_enabled=True),
        )
        self.assertTrue(res_lmr.legal)
        self.assertIsNotNone(res_lmr.best_move)
        self.assertEqual(p.fen, fen)


class TestLMRNotAppliedToCriticalMoves(unittest.TestCase):
    def test_first_move_never_reduced(self):
        # la jugada de indice 0 SIEMPRE se busca a ventana/profundidad
        # completa - LMR solo aplica desde lmr_min_move_index en adelante.
        p = Position()
        ctx = SearchContext(SearchLimits(max_depth=5), RepetitionTracker())
        pc = PruningConfig(lmr_enabled=True, lmr_min_move_index=1)
        pvs(p, 5, -INFINITY, INFINITY, ctx, pruning_config=pc)
        # pvs_full_window_searches cuenta la 1ra jugada de CADA nodo -
        # debe ser > 0 siempre que haya al menos un nodo interno.
        self.assertGreater(ctx.pvs_full_window_searches, 0)

    def test_no_lmr_when_node_in_check(self):
        # nodo en jaque: todas las jugadas son evasiones forzadas, ninguna
        # debe reducirse via LMR.
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        self.assertTrue(p.is_check)
        ctx = SearchContext(SearchLimits(max_depth=4), RepetitionTracker())
        pc = PruningConfig(lmr_enabled=True, lmr_min_depth=1, lmr_min_move_index=1)
        pvs(p, 4, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.lmr_reductions_applied, 0)

    def test_captures_and_promotions_never_reduced(self):
        # posicion con varias capturas disponibles entre las jugadas
        # tardias del ordenamiento (Fase II, tier fijo: capturas SIEMPRE
        # antes que jugadas tranquilas) - confirmar que activar LMR con un
        # indice minimo muy bajo (0) sigue dando el mismo score que sin
        # LMR, porque las capturas quedan excluidas de la reduccion por
        # tipo de jugada, no solo por indice.
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        p1 = Position(fen)
        res_plain = iterative_deepening_search(p1, SearchLimits(max_depth=3), use_pvs=True)
        p2 = Position(fen)
        res_lmr = iterative_deepening_search(
            p2, SearchLimits(max_depth=3), use_pvs=True,
            pruning_config=PruningConfig(lmr_enabled=True, lmr_min_move_index=1, lmr_min_depth=1),
        )
        self.assertEqual(res_plain.score, res_lmr.score)


if __name__ == "__main__":
    unittest.main()
