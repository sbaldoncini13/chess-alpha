"""Tests de null move pruning (Fase V, items 29-30): nunca se intenta
estando en jaque, guarda de zugzwang (finales de solo peones/rey) - mismo
patron ya usado para verificar guardas de seguridad en Fase IV
(quiescence: nunca stand-pat en jaque).

Correr: python -m pytest tests/search_v1/test_null_move.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY, Position, PruningConfig, RepetitionTracker,
    SearchContext, SearchLimits, iterative_deepening_search, pvs,
)


class TestNullMoveForbiddenInCheck(unittest.TestCase):
    def test_no_null_move_attempt_when_in_check(self):
        # depth=1 con null_move_min_depth=1: el UNICO nodo con profundidad
        # suficiente para siquiera evaluar la guarda de null move es la
        # raiz (los hijos quedan a depth=0, fuera del alcance de la
        # tecnica). `ply_from_root` se fuerza a 1 para satisfacer la
        # guarda de "no en la raiz" y aislar especificamente la guarda de
        # jaque (ver test_null_move_not_attempted_at_root abajo para esa
        # otra guarda per se).
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"  # torre negra da jaque a blancas
        p = Position(fen)
        self.assertTrue(p.is_check)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        ctx.ply_from_root = 1
        pc = PruningConfig(null_move_enabled=True, null_move_min_depth=1)
        score, pv = pvs(p, 1, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.null_move_attempts, 0)  # nunca se intento estando en jaque

    def test_null_move_not_attempted_at_root(self):
        # guarda separada (documentada explicitamente en pvs.py): null
        # move NUNCA se intenta en el nodo raiz de la busqueda, este de
        # jaque o no - confirmado con una posicion SIN jaque.
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        self.assertFalse(p.is_check)
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        pc = PruningConfig(null_move_enabled=True, null_move_min_depth=1)
        pvs(p, 1, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.null_move_attempts, 0)

    def test_apply_null_would_be_illegal_if_misused_directly(self):
        # invariante de seguridad de Position.apply_null(): documentado
        # como responsabilidad del LLAMADOR nunca invocarlo en jaque - se
        # verifica aca que efectivamente NO hay guarda automatica adentro
        # de Position (la guarda vive en pvs.py, no duplicada), y que el
        # motor real (pvs con pruning_config) respeta esa responsabilidad
        # (ver test de arriba: null_move_attempts=0 con el nodo raiz en jaque).
        fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
        p = Position(fen)
        # aplicar null estando en jaque no lanza IllegalMoveError (no pasa
        # por is_legal) - por eso pvs.py DEBE guardarlo explicitamente,
        # verificado en el test anterior.
        p.apply_null()
        p.undo()
        self.assertEqual(p.fen, fen)


class TestNullMoveZugzwangGuard(unittest.TestCase):
    def test_no_null_move_without_non_pawn_material(self):
        # final de solo peones y reyes - null move pruning NUNCA debe
        # intentarse aca (zugzwang tipico: "pasar" cambiaria el resultado).
        fen = "8/4k3/8/3pP3/8/8/4K3/8 w - - 0 1"
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=4), RepetitionTracker())
        pc = PruningConfig(null_move_enabled=True, null_move_min_depth=1)
        score, pv = pvs(p, 4, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertEqual(ctx.null_move_attempts, 0)

    def test_score_matches_reference_in_pawn_endgame_with_null_move_enabled(self):
        # aunque la guarda ya evita el intento, se verifica ademas que el
        # SCORE final en este final de peones coincide con la referencia
        # sin podas - doble candado (guarda + resultado correcto).
        fen = "8/4k3/8/3pP3/8/8/4K3/8 w - - 0 1"
        p1 = Position(fen)
        res_plain = iterative_deepening_search(p1, SearchLimits(max_depth=4), use_pvs=True)
        p2 = Position(fen)
        res_null = iterative_deepening_search(
            p2, SearchLimits(max_depth=4), use_pvs=True,
            pruning_config=PruningConfig(null_move_enabled=True, null_move_min_depth=1),
        )
        self.assertEqual(res_plain.score, res_null.score)

    def test_null_move_attempted_with_non_pawn_material_present(self):
        # control positivo: CON piezas mayores/menores, el nodo si intenta
        # null move (confirma que la guarda de arriba es especifica de
        # material, no un bug que desactiva null move siempre).
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        ctx = SearchContext(SearchLimits(max_depth=4), RepetitionTracker())
        pc = PruningConfig(null_move_enabled=True, null_move_min_depth=1)
        pvs(p, 4, -INFINITY, INFINITY, ctx, pruning_config=pc)
        self.assertGreater(ctx.null_move_attempts, 0)


class TestNullMoveRootIntact(unittest.TestCase):
    def test_root_intact_with_null_move_enabled(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        p = Position(fen)
        iterative_deepening_search(
            p, SearchLimits(max_depth=4), use_pvs=True,
            pruning_config=PruningConfig(null_move_enabled=True),
        )
        self.assertEqual(p.fen, fen)


if __name__ == "__main__":
    unittest.main()
