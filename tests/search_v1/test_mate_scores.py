"""Tests de la convencion de mate scores (Fase II).

Correr: python -m pytest tests/search_v1/test_mate_scores.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import MATE_SCORE, is_mate_score, mate_distance_plies_from_score, score_from_tt, score_to_tt
from search_v1.static_evaluation import PIECE_VALUES_CP


class TestMateScoreThreshold(unittest.TestCase):
    def test_normal_cp_scores_not_mate(self):
        for score in [0, 100, -100, 900, -900, PIECE_VALUES_CP[5] * 9]:  # margen generoso
            with self.subTest(score=score):
                self.assertFalse(is_mate_score(score))

    def test_mate_scores_detected(self):
        self.assertTrue(is_mate_score(MATE_SCORE))
        self.assertTrue(is_mate_score(MATE_SCORE - 1))
        self.assertTrue(is_mate_score(-(MATE_SCORE - 1)))

    def test_mate_score_never_overflows_relative_to_material(self):
        # ni siquiera un ejercito completo de 8 peones promovidos a dama
        # por bando se acerca al umbral de mate
        extreme_material = 900 * 9 * 2  # generoso
        self.assertLess(extreme_material, MATE_SCORE - 1000)


class TestMateDistance(unittest.TestCase):
    def test_shorter_mate_scores_higher_when_favorable(self):
        # preferencia por mate mas corto: a favor, ply menor -> score mayor
        score_mate_in_1_ply = MATE_SCORE - 1
        score_mate_in_3_ply = MATE_SCORE - 3
        self.assertGreater(score_mate_in_1_ply, score_mate_in_3_ply)

    def test_longer_forced_loss_scores_higher_when_unfavorable(self):
        # retraso de mate inevitable: en contra, ply mayor (mas lejos) ->
        # score MENOS negativo (mayor) - se prefiere retrasar lo inevitable
        score_mated_in_1_ply = -(MATE_SCORE - 1)
        score_mated_in_3_ply = -(MATE_SCORE - 3)
        self.assertGreater(score_mated_in_3_ply, score_mated_in_1_ply)

    def test_mate_distance_positive_when_favorable(self):
        self.assertEqual(mate_distance_plies_from_score(MATE_SCORE - 1), 1)
        self.assertEqual(mate_distance_plies_from_score(MATE_SCORE - 5), 5)

    def test_mate_distance_negative_when_unfavorable(self):
        self.assertEqual(mate_distance_plies_from_score(-(MATE_SCORE - 1)), -1)
        self.assertEqual(mate_distance_plies_from_score(-(MATE_SCORE - 5)), -5)

    def test_mate_distance_none_for_cp_score(self):
        self.assertIsNone(mate_distance_plies_from_score(0))
        self.assertIsNone(mate_distance_plies_from_score(900))
        self.assertIsNone(mate_distance_plies_from_score(-900))


class TestTTScoreNormalization(unittest.TestCase):
    """Items 14-16 de Fase III: normalizacion de mate scores para la TT."""

    def test_cp_score_unaffected_by_normalization(self):
        for ply in (0, 1, 5):
            with self.subTest(ply=ply):
                self.assertEqual(score_to_tt(150, ply), 150)
                self.assertEqual(score_from_tt(150, ply), 150)

    def test_positive_mate_stored_and_recovered_same_ply(self):
        # mate en 1 ply guardado y recuperado en el mismo ply (0) - sin cambio
        score = MATE_SCORE - 1
        stored = score_to_tt(score, ply_from_root=0)
        self.assertEqual(score_from_tt(stored, ply_from_root=0), score)

    def test_positive_mate_recovered_from_different_ply(self):
        # se guarda un mate-en-1-ply-desde-el-nodo en un nodo que esta a
        # ply_from_root=2 de la raiz ORIGINAL; una busqueda DISTINTA que
        # llega a la MISMA posicion a ply_from_root=5 debe recuperar
        # "mate en 1 ply desde aca" = mate en 6 ply desde SU propia raiz
        score_at_storage = MATE_SCORE - 3  # "mate en 3 ply" visto desde la raiz que lo guardo (ply_from_root=2 en ese momento)
        stored = score_to_tt(score_at_storage, ply_from_root=2)  # normaliza a "1 ply desde el nodo"
        self.assertEqual(stored, MATE_SCORE - 1)
        recovered = score_from_tt(stored, ply_from_root=5)  # otra busqueda, llega a ply 5
        self.assertEqual(recovered, MATE_SCORE - 6)  # 5 + 1 = mate en 6 ply desde ESA raiz

    def test_negative_mate_recovered_from_different_ply(self):
        score_at_storage = -(MATE_SCORE - 5)  # "mate en contra a 5 ply" visto desde ply_from_root=2
        stored = score_to_tt(score_at_storage, ply_from_root=2)  # normaliza a "3 ply desde el nodo"
        self.assertEqual(stored, -(MATE_SCORE - 3))
        recovered = score_from_tt(stored, ply_from_root=4)  # otra busqueda, llega a ply 4
        self.assertEqual(recovered, -(MATE_SCORE - 7))  # 4 + 3 = mate en contra a 7 ply desde ESA raiz

    def test_roundtrip_identity_at_same_ply(self):
        for score in (MATE_SCORE - 1, MATE_SCORE - 10, -(MATE_SCORE - 1), -(MATE_SCORE - 10), 0, 250, -250):
            for ply in (0, 3, 7):
                with self.subTest(score=score, ply=ply):
                    self.assertEqual(score_from_tt(score_to_tt(score, ply), ply), score)

    def test_shorter_mate_still_preferred_after_normalization(self):
        # preferencia por mate mas corto conservada tras normalizar/recuperar
        stored_short = score_to_tt(MATE_SCORE - 1, ply_from_root=0)
        stored_long = score_to_tt(MATE_SCORE - 3, ply_from_root=0)
        recovered_short = score_from_tt(stored_short, ply_from_root=0)
        recovered_long = score_from_tt(stored_long, ply_from_root=0)
        self.assertGreater(recovered_short, recovered_long)

    def test_delay_of_inevitable_mate_conserved_after_normalization(self):
        stored_fast_loss = score_to_tt(-(MATE_SCORE - 1), ply_from_root=0)
        stored_slow_loss = score_to_tt(-(MATE_SCORE - 3), ply_from_root=0)
        recovered_fast = score_from_tt(stored_fast_loss, ply_from_root=0)
        recovered_slow = score_from_tt(stored_slow_loss, ply_from_root=0)
        self.assertGreater(recovered_slow, recovered_fast)


if __name__ == "__main__":
    unittest.main()
