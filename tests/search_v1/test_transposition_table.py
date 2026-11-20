"""Tests de la Transposition Table (Fase III): store/probe, bounds,
colisiones, reemplazo, generaciones, capacidad/ocupacion.

Correr: python -m pytest tests/search_v1/test_transposition_table.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import Bound, TranspositionTable


class TestStoreProbeBasic(unittest.TestCase):
    def test_store_then_probe_hit(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(key=100, depth=3, score=50, bound=Bound.EXACT, best_move="e2e4")
        entry = tt.probe(100)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.score, 50)
        self.assertEqual(entry.best_move, "e2e4")

    def test_probe_miss_empty(self):
        tt = TranspositionTable(capacity_entries=16)
        self.assertIsNone(tt.probe(999))

    def test_probe_increments_counters(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(key=1, depth=2, score=0, bound=Bound.EXACT, best_move=None)
        tt.probe(1)
        tt.probe(2)
        self.assertEqual(tt.probes, 2)
        self.assertEqual(tt.hits, 1)


class TestBounds(unittest.TestCase):
    def test_exact_bound_stored_and_retrieved(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, 3, 120, Bound.EXACT, "e2e4")
        entry = tt.probe(1)
        self.assertEqual(entry.bound, Bound.EXACT)
        self.assertEqual(tt.exact_hits, 1)

    def test_lower_bound_stored_and_retrieved(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, 3, 120, Bound.LOWER_BOUND, "e2e4")
        entry = tt.probe(1)
        self.assertEqual(entry.bound, Bound.LOWER_BOUND)
        self.assertEqual(tt.lower_hits, 1)

    def test_upper_bound_stored_and_retrieved(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, 3, 120, Bound.UPPER_BOUND, "e2e4")
        entry = tt.probe(1)
        self.assertEqual(entry.bound, Bound.UPPER_BOUND)
        self.assertEqual(tt.upper_hits, 1)


class TestDepthSufficiency(unittest.TestCase):
    def test_sufficient_depth_usable(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, depth=5, score=0, bound=Bound.EXACT, best_move=None)
        entry = tt.probe(1, min_depth=3)
        self.assertIsNotNone(entry)

    def test_insufficient_depth_rejected(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, depth=2, score=0, bound=Bound.EXACT, best_move=None)
        entry = tt.probe(1, min_depth=5)
        self.assertIsNone(entry)
        # no cuenta como hit real si la profundidad es insuficiente
        self.assertEqual(tt.hits, 0)


class TestCollisions(unittest.TestCase):
    def test_artificial_index_collision_detected(self):
        # capacity=1 fuerza a que CUALQUIER clave distinta comparta el
        # mismo indice (0) - la clave completa debe evitar el falso hit
        tt = TranspositionTable(capacity_entries=1)
        tt.store(key=111, depth=3, score=10, bound=Bound.EXACT, best_move="e2e4")
        entry = tt.probe(key=222)  # mismo indice (0), clave distinta
        self.assertIsNone(entry)
        self.assertEqual(tt.collisions, 1)

    def test_full_key_prevents_false_hit(self):
        # colision de indice real (capacity=1): la 2da entrada tiene mas
        # profundidad, asi que SI reemplaza (politica normal de
        # reemplazo aplicada tambien a colisiones de indice) - lo que se
        # verifica es que el valor recuperado es el de 222 (correcto,
        # sin mezcla de datos de 111) y no un hit espurio de 111 bajo la
        # clave de 222.
        tt = TranspositionTable(capacity_entries=1)
        tt.store(key=111, depth=3, score=10, bound=Bound.EXACT, best_move="e2e4")
        tt.store(key=222, depth=5, score=99, bound=Bound.EXACT, best_move="d2d4")
        entry = tt.probe(key=222)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.score, 99)  # la entrada de 111 fue reemplazada, no mezclada
        self.assertIsNone(tt.probe(key=111))  # 111 ya no esta (se libero el unico bucket)

    def test_same_key_no_collision(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(key=555, depth=3, score=10, bound=Bound.EXACT, best_move="e2e4")
        tt.probe(key=555)
        self.assertEqual(tt.collisions, 0)


class TestReplacementStrategy(unittest.TestCase):
    def test_replace_empty_bucket(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, 3, 0, Bound.EXACT, None)
        self.assertEqual(tt.replacements, 0)  # bucket vacio no cuenta como "reemplazo"
        self.assertIsNotNone(tt.probe(1))

    def test_replace_by_greater_depth(self):
        tt = TranspositionTable(capacity_entries=1)
        tt.store(1, depth=2, score=10, bound=Bound.EXACT, best_move="a")
        tt.store(1, depth=5, score=20, bound=Bound.EXACT, best_move="b")
        entry = tt.probe(1)
        self.assertEqual(entry.depth, 5)
        self.assertEqual(entry.best_move, "b")
        self.assertEqual(tt.replacements, 1)

    def test_no_replace_by_lesser_depth(self):
        tt = TranspositionTable(capacity_entries=1)
        tt.store(1, depth=5, score=10, bound=Bound.EXACT, best_move="a")
        tt.store(1, depth=2, score=20, bound=Bound.EXACT, best_move="b")
        entry = tt.probe(1)
        self.assertEqual(entry.depth, 5)
        self.assertEqual(entry.best_move, "a")  # conservada, no reemplazada

    def test_exact_replaces_bound_at_equal_depth(self):
        tt = TranspositionTable(capacity_entries=1)
        tt.store(1, depth=3, score=10, bound=Bound.LOWER_BOUND, best_move="a")
        tt.store(1, depth=3, score=20, bound=Bound.EXACT, best_move="b")
        entry = tt.probe(1)
        self.assertEqual(entry.bound, Bound.EXACT)
        self.assertEqual(entry.best_move, "b")

    def test_bound_does_not_replace_exact_at_equal_depth(self):
        tt = TranspositionTable(capacity_entries=1)
        tt.store(1, depth=3, score=10, bound=Bound.EXACT, best_move="a")
        tt.store(1, depth=3, score=20, bound=Bound.LOWER_BOUND, best_move="b")
        entry = tt.probe(1)
        self.assertEqual(entry.bound, Bound.EXACT)
        self.assertEqual(entry.best_move, "a")  # conservada

    def test_replace_by_older_generation(self):
        tt = TranspositionTable(capacity_entries=1)
        tt.store(1, depth=5, score=10, bound=Bound.EXACT, best_move="a")
        tt.new_generation()
        tt.store(2, depth=1, score=99, bound=Bound.UPPER_BOUND, best_move="b")  # colisiona de indice con key=1
        entry = tt.probe(2)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.best_move, "b")  # generacion vieja se reemplaza SIEMPRE, aunque tenga mas profundidad


class TestGenerationsAndMaintenance(unittest.TestCase):
    def test_new_generation_increments(self):
        tt = TranspositionTable(capacity_entries=16)
        self.assertEqual(tt.generation, 0)
        tt.new_generation()
        self.assertEqual(tt.generation, 1)

    def test_clear_resets_everything(self):
        tt = TranspositionTable(capacity_entries=16)
        tt.store(1, 3, 10, Bound.EXACT, "a")
        tt.probe(1)
        tt.new_generation()
        tt.clear()
        self.assertIsNone(tt.probe(1))
        self.assertEqual(tt.generation, 0)
        self.assertEqual(tt.stores, 0)
        self.assertEqual(tt.occupancy(), 0)


class TestCapacityAndOccupancy(unittest.TestCase):
    def test_occupancy_counts_filled_buckets(self):
        tt = TranspositionTable(capacity_entries=16)
        self.assertEqual(tt.occupancy(), 0)
        tt.store(1, 3, 0, Bound.EXACT, None)
        tt.store(2, 3, 0, Bound.EXACT, None)
        self.assertEqual(tt.occupancy(), 2)

    def test_capacity_and_estimated_bytes(self):
        tt = TranspositionTable(capacity_entries=1000)
        self.assertEqual(tt.capacity, 1000)
        self.assertGreater(tt.estimated_bytes(), 0)

    def test_small_capacity_for_tests(self):
        # tamanos chicos y controlados, tal como pide el pedido
        tt = TranspositionTable(capacity_entries=64)
        self.assertEqual(tt.capacity, 64)

    def test_invalid_capacity_rejected(self):
        with self.assertRaises(ValueError):
            TranspositionTable(capacity_entries=0)


if __name__ == "__main__":
    unittest.main()
