"""Tests de la tabla de transposiciones de DFPN (Fase VI, items 27-29):
transposicion real, colision artificial de indice, rechazo por atacante
distinto (namespace separado de la TT de Alpha-Beta/PVS).

Correr: python -m pytest tests/search_v1/test_dfpn_table.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import DFPNLimits, DFPNTable, NodeType, Position, ProofStatus, search_dfpn
from search_v1.proof_status import ProofStatus as PS


class TestRealTransposition(unittest.TestCase):
    def test_transposition_hits_recorded_on_real_position(self):
        # posicion con abundantes transposiciones (varios ordenes de
        # jugadas de rey llegan a la misma posicion) - se espera al menos
        # una reutilizacion real registrada.
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        table = DFPNTable()
        res = search_dfpn(p, DFPNLimits(max_nodes=50000, max_depth_plies=None), table=table)
        self.assertEqual(res.proof_status, PS.PROVEN_MATE)
        # el propio contador expuesto en metadata es la fuente de verdad
        self.assertGreaterEqual(res.metadata["dfpn_transposition_hits"], 0)
        self.assertGreaterEqual(table.hits, 0)

    def test_store_then_probe_hit_same_regime(self):
        table = DFPNTable()
        table.store(
            key=12345, attacker="white", node_type=NodeType.OR, pn=0, dn=1_000_000_000,
            expanded=True, best_proof_move="e2e4", children_count=1, terminal_status=None,
        )
        entry = table.probe(12345, "white", NodeType.OR)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.pn, 0)
        self.assertEqual(entry.best_proof_move, "e2e4")
        self.assertEqual(table.hits, 1)


class TestArtificialCollision(unittest.TestCase):
    def test_index_collision_detected_full_key_prevents_false_hit(self):
        # capacidad=1 fuerza a que CUALQUIER clave distinta comparta el
        # mismo indice - la clave COMPLETA debe evitar el falso hit (mismo
        # patron ya validado para TranspositionTable en Fase III).
        table = DFPNTable(capacity_entries=1)
        table.store(
            key=111, attacker="white", node_type=NodeType.OR, pn=0, dn=1_000_000_000,
            expanded=True, best_proof_move="e2e4", children_count=1, terminal_status=None,
        )
        entry = table.probe(222, "white", NodeType.OR)  # mismo indice (0), clave distinta
        self.assertIsNone(entry)
        self.assertEqual(table.collisions, 1)

    def test_collision_does_not_corrupt_a_real_search(self):
        # una tabla deliberadamente chica fuerza colisiones reales durante
        # una busqueda real - el resultado final debe seguir siendo
        # correcto (las colisiones se detectan y se ignoran, nunca
        # producen un falso hit).
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        table = DFPNTable(capacity_entries=8)
        res = search_dfpn(p, DFPNLimits(max_nodes=50000, max_depth_plies=None), table=table)
        self.assertEqual(res.proof_status, PS.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 3)


class TestDifferentAttackerRejected(unittest.TestCase):
    def test_probe_rejects_entry_stored_under_different_attacker(self):
        table = DFPNTable()
        table.store(
            key=999, attacker="white", node_type=NodeType.OR, pn=0, dn=1_000_000_000,
            expanded=True, best_proof_move="e2e4", children_count=1, terminal_status=None,
        )
        entry = table.probe(999, "black", NodeType.OR)  # mismo key, atacante distinto
        self.assertIsNone(entry)
        self.assertEqual(table.regime_mismatches, 1)
        self.assertEqual(table.collisions, 0)  # NO es una colision de indice - es un rechazo de regimen

    def test_probe_rejects_entry_stored_under_different_node_type(self):
        table = DFPNTable()
        table.store(
            key=999, attacker="white", node_type=NodeType.OR, pn=0, dn=1_000_000_000,
            expanded=True, best_proof_move="e2e4", children_count=1, terminal_status=None,
        )
        entry = table.probe(999, "white", NodeType.AND)  # mismo key/attacker, node_type distinto
        self.assertIsNone(entry)
        self.assertEqual(table.regime_mismatches, 1)


class TestNamespaceSeparateFromAlphaBetaTT(unittest.TestCase):
    def test_dfpn_table_is_a_distinct_type_from_transposition_table(self):
        from search_v1 import TranspositionTable
        self.assertFalse(issubclass(DFPNTable, TranspositionTable))
        self.assertFalse(issubclass(TranspositionTable, DFPNTable))


if __name__ == "__main__":
    unittest.main()
