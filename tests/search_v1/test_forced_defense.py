"""Tests de defensa obligatoria (Fase VI, items 16-19): defensa unica, dos
defensas equivalentes, ninguna defensa, rama UNKNOWN impide declarar
defensa unica.

Fixtures verificados empiricamente (K+Q vs K, definicion adoptada en
forced_defense.py: "preserva" = el oponente NO fuerza mate probado via
DFPN tras esa jugada, dentro del presupuesto dado):
- Ninguna defensa: "k7/8/2K5/8/8/8/8/7Q b" - las 2 jugadas legales de
  negras llevan a mate forzado de blancas, confirmado con presupuesto
  amplio (sin UNKNOWN).
- Dos defensas: "3k4/8/8/2K5/8/8/8/6Q1 b" a profundidad acotada 6 - de
  las 5 jugadas legales, 2 resisten mas alla de la cota (preservan bajo
  esta definicion) y 3 caen en mate forzado dentro de la cota.
- Defensa unica: mismo patron + un peon blanco en d6 (quita c7/e7 de las
  jugadas legales sin agregar poder de mate significativo) - de las 3
  jugadas legales que quedan, solo d8d7 preserva.

Correr: python -m pytest tests/search_v1/test_forced_defense.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import DFPNLimits, Position, analyze_forced_defense

_NO_DEFENSE_FEN = "k7/8/2K5/8/8/8/8/7Q b - - 0 1"
_TWO_DEFENSES_FEN = "3k4/8/8/2K5/8/8/8/6Q1 b - - 0 1"
_UNIQUE_DEFENSE_FEN = "3k4/8/3P4/2K5/8/8/8/6Q1 b - - 0 1"


class TestUniqueDefense(unittest.TestCase):
    def test_exactly_one_preserving_move(self):
        p = Position(_UNIQUE_DEFENSE_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=200_000, max_depth_plies=6))
        self.assertEqual(res.legal_moves_count, 3)
        self.assertEqual(res.preserving_moves, ["d8d7"])
        self.assertEqual(sorted(res.losing_moves), ["d8c8", "d8e8"])
        self.assertEqual(res.unknown_moves, [])
        self.assertTrue(res.unique_forced_defense)

    def test_root_intact(self):
        p = Position(_UNIQUE_DEFENSE_FEN)
        analyze_forced_defense(p, DFPNLimits(max_nodes=200_000, max_depth_plies=6))
        self.assertEqual(p.fen, _UNIQUE_DEFENSE_FEN)


class TestNoDefense(unittest.TestCase):
    def test_all_legal_moves_lose(self):
        p = Position(_NO_DEFENSE_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=5000, max_depth_plies=8))
        self.assertEqual(res.legal_moves_count, 2)
        self.assertEqual(res.preserving_moves, [])
        self.assertEqual(sorted(res.losing_moves), ["a8a7", "a8b8"])
        self.assertEqual(res.unknown_moves, [])
        self.assertFalse(res.unique_forced_defense)

    def test_root_intact(self):
        p = Position(_NO_DEFENSE_FEN)
        analyze_forced_defense(p, DFPNLimits(max_nodes=5000, max_depth_plies=8))
        self.assertEqual(p.fen, _NO_DEFENSE_FEN)


class TestTwoEquivalentDefenses(unittest.TestCase):
    def test_exactly_two_preserving_moves(self):
        p = Position(_TWO_DEFENSES_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=200_000, max_depth_plies=6))
        self.assertEqual(res.legal_moves_count, 5)
        self.assertEqual(sorted(res.preserving_moves), ["d8d7", "d8e7"])
        self.assertEqual(len(res.losing_moves), 3)
        self.assertEqual(res.unknown_moves, [])
        # 2 preservadoras -> NUNCA se declara defensa unica
        self.assertFalse(res.unique_forced_defense)


class TestUnknownBlocksUniqueDeclaration(unittest.TestCase):
    def test_unknown_branch_prevents_unique_declaration(self):
        # MISMA posicion que arriba, presupuesto deliberadamente chico -
        # algunas jugadas quedan UNKNOWN (no se completo la prueba) - aun
        # si el resto "parece" apuntar a una unica preservadora,
        # unique_forced_defense DEBE quedar False mientras haya CUALQUIER
        # rama sin resolver (item 4 de la definicion en forced_defense.py).
        p = Position(_TWO_DEFENSES_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=3000, max_depth_plies=6))
        self.assertGreater(len(res.unknown_moves), 0)
        self.assertFalse(res.unique_forced_defense)


class TestForcedDefenseStructure(unittest.TestCase):
    def test_move_classifications_cover_all_legal_moves(self):
        p = Position(_NO_DEFENSE_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=5000, max_depth_plies=8))
        self.assertEqual(len(res.move_classifications), res.legal_moves_count)
        classified_ucis = {c.move_uci for c in res.move_classifications}
        self.assertEqual(
            classified_ucis,
            set(res.preserving_moves) | set(res.losing_moves) | set(res.unknown_moves),
        )

    def test_losing_moves_have_refutation_pv(self):
        p = Position(_NO_DEFENSE_FEN)
        res = analyze_forced_defense(p, DFPNLimits(max_nodes=5000, max_depth_plies=8))
        for c in res.move_classifications:
            if c.classification == "losing":
                self.assertGreater(len(c.refutation_pv), 0)
                self.assertIsNotNone(c.mate_distance_plies)


if __name__ == "__main__":
    unittest.main()
