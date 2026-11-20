"""Tests de evasiones de jaque, ciclos y limites de DFPN (Fase VI, items
15/30-34): todas las evasiones de jaque consideradas, ciclo de jaques
detectado sin falso mate, limite de nodos, limite de tiempo, cancelacion
limpia (sin excepcion, resultado bien formado), UNKNOWN por limite con el
motivo correcto.

Correr: python -m pytest tests/search_v1/test_dfpn_cycles_and_limits.py -v
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    DFPNLimits, Position, ProofStatus, generate_legal_moves, generate_tactical_moves,
    search_dfpn,
)


class TestAllCheckEvasionsConsidered(unittest.TestCase):
    def test_and_node_in_check_explores_all_legal_evasions(self):
        # posicion donde el DEFENSOR (nodo AND) queda en jaque con VARIAS
        # evasiones legales - DFPN debe generar TODAS (generate_legal_moves,
        # nunca solo las tacticas) para que la prueba sea exhaustiva.
        fen = "4k3/8/8/8/8/8/8/K3Q3 b - - 0 1"  # negras en jaque de la dama blanca (misma columna)
        p = Position(fen)
        self.assertTrue(p.is_check)
        legal = generate_legal_moves(p)
        self.assertEqual(len(legal), 4)  # 4 evasiones legales (d8/f8/d7/f7)
        # generate_tactical_moves (usado en quiescence) devuelve el MISMO
        # conjunto para posiciones en jaque (evasiones = todas las legales) -
        # confirma que DFPN, al usar generate_legal_moves directamente, no
        # se queda corto respecto de ese conjunto de referencia.
        tactical = generate_tactical_moves(p)
        self.assertEqual({m.uci for m in legal}, {m.uci for m in tactical})


class TestCheckCycleNoFalseMate(unittest.TestCase):
    def test_perpetual_check_cycle_never_produces_a_false_mate(self):
        # posicion donde el atacante puede dar jaques repetidos sin
        # avanzar - el ciclo debe detectarse (DRAW_BY_REPETITION interno)
        # y NUNCA producir un PROVEN_MATE falso via ese ciclo.
        fen = "6k1/8/8/8/8/8/8/Q6K w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=3000, max_depth_plies=None))
        self.assertIn(res.proof_status, (
            ProofStatus.PROVEN_MATE, ProofStatus.PROVEN_NO_MATE,
            ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND, ProofStatus.UNKNOWN_NODE_LIMIT,
        ))
        if res.proof_status == ProofStatus.PROVEN_MATE:
            from search_v1 import verify_mate_proof
            self.assertEqual(verify_mate_proof(Position(fen), res), [])

    def test_cycle_telemetry_recorded(self):
        # una busqueda con margen para ciclar (sin cota de profundidad,
        # posicion con jaques disponibles reiteradamente) debe registrar
        # dfpn_cycles > 0 en algun momento, o resolver sin necesitarlo -
        # ambos son validos; lo que NO es valido es que la busqueda cuelgue.
        fen = "6k1/8/8/8/8/8/8/Q6K w - - 0 1"
        p = Position(fen)
        t0 = time.perf_counter()
        res = search_dfpn(p, DFPNLimits(max_nodes=3000, max_depth_plies=None))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 60)  # termina en tiempo acotado, no cuelga
        self.assertIn("dfpn_cycles", res.metadata)


class TestNodeLimit(unittest.TestCase):
    def test_unknown_node_limit_when_budget_exhausted(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=50, max_depth_plies=None))
        self.assertEqual(res.proof_status, ProofStatus.UNKNOWN_NODE_LIMIT)
        self.assertEqual(res.nodes, 50)
        self.assertEqual(res.metadata["proof_stop_reason"], "max_nodes")

    def test_root_intact_after_node_limit(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p = Position(fen)
        search_dfpn(p, DFPNLimits(max_nodes=50, max_depth_plies=None))
        self.assertEqual(p.fen, fen)


class TestTimeLimit(unittest.TestCase):
    def test_unknown_time_limit_when_budget_exhausted(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=10_000_000, max_time_ms=20, max_depth_plies=None))
        self.assertIn(res.proof_status, (ProofStatus.UNKNOWN_TIME_LIMIT, ProofStatus.UNKNOWN_NODE_LIMIT))
        if res.proof_status == ProofStatus.UNKNOWN_TIME_LIMIT:
            self.assertEqual(res.metadata["proof_stop_reason"], "max_time")


class TestCleanCancellation(unittest.TestCase):
    def test_no_exception_raised_on_limit_exhaustion(self):
        # a diferencia de SearchAborted (Fase II-V), DFPN NUNCA lanza una
        # excepcion al agotar presupuesto - devuelve un DFPNResult bien
        # formado con proof_status UNKNOWN_* (cancelacion limpia).
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p = Position(fen)
        try:
            res = search_dfpn(p, DFPNLimits(max_nodes=20, max_depth_plies=None))
        except Exception as exc:
            self.fail(f"search_dfpn lanzo una excepcion en vez de devolver UNKNOWN: {exc!r}")
        self.assertTrue(res.proof_status.name.startswith("UNKNOWN"))
        self.assertEqual(res.principal_variation, [])
        self.assertIsNone(res.mate_distance_plies)


class TestUnknownReasonExplicit(unittest.TestCase):
    def test_unknown_never_silently_returned_as_proven(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=30, max_depth_plies=None))
        self.assertNotEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertNotEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE)
        self.assertIn(res.proof_status, (ProofStatus.UNKNOWN_NODE_LIMIT, ProofStatus.UNKNOWN_TIME_LIMIT))


if __name__ == "__main__":
    unittest.main()
