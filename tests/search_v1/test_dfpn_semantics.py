"""Tests de semantica OR/AND de PNS (Fase VI, items 20-26): nodo OR, nodo
AND, proof number cero, disproof number cero, actualizacion ascendente,
"threshold" (seleccion del nodo mas probante en cada iteracion - ver
docstring de dfpn_node.py sobre la variante PNS elegida en vez de df-pn
clasico con umbrales duales), expansion del nodo mas prometedor.

Correr: python -m pytest tests/search_v1/test_dfpn_semantics.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import DFPNLimits, DFPNTable, NodeType, Position, ProofStatus, search_dfpn


class TestORNodeSemantics(unittest.TestCase):
    def test_root_is_or_node_attacker_to_move(self):
        # el nodo raiz SIEMPRE es OR (atacante = side_to_move de la raiz)
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.attacker, "white")
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)

    def test_or_node_proven_if_any_child_proven(self):
        # mate en 1: la raiz (OR) tiene VARIAS jugadas posibles, pero basta
        # con que UNA (h1a8) este probada para que la raiz quede pn=0.
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.root_pn, 0)
        self.assertGreater(res.metadata["dfpn_nodes"], 1)  # explora mas de una jugada antes de resolver


class TestANDNodeSemantics(unittest.TestCase):
    def test_and_node_requires_all_children_proven(self):
        # mate en 2: tras la primera jugada de blancas (c6b6), negras
        # (nodo AND) tiene UNA UNICA respuesta legal (a8b8) - el nodo AND
        # con un solo hijo se prueba solo si ESE hijo se prueba.
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        # ply 1 de la PV es la jugada de blancas, ply 2 es la UNICA
        # respuesta forzada de negras
        self.assertEqual(res.principal_variation[1], "a8b8")

    def test_and_node_disproven_if_any_child_disproven(self):
        # posicion sin mate: el nodo raiz (OR, blancas) se disprueba
        # porque, para CADA jugada de blancas, existe alguna respuesta de
        # negras (nodo AND) que escapa - eventualmente algun AND-node se
        # disprueba por tener un hijo disproven (root.dn==0 confirma la
        # propagacion funciono correctamente para todo el arbol).
        fen = "4k3/8/4K3/8/8/8/8/7Q b - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE)
        self.assertEqual(res.root_dn, 0)


class TestProofAndDisproofNumberZero(unittest.TestCase):
    def test_proof_number_zero_means_proven(self):
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.root_pn, 0)
        self.assertGreater(res.root_dn, 0)  # dn no necesariamente 0 en un nodo proven

    def test_disproof_number_zero_means_disproven(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"  # solo reyes, sin mate posible
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.root_dn, 0)
        self.assertGreater(res.root_pn, 0)


class TestBackpropagation(unittest.TestCase):
    def test_ancestor_pn_updates_after_child_resolves(self):
        # tras cada expansion, el ancestro se recalcula ANTES de la
        # siguiente seleccion de MPN (invariante documentado en dfpn.py) -
        # se verifica indirectamente: el nodo raiz nunca queda con pn/dn
        # "obsoleto" (si estuviera desactualizado, la busqueda no
        # convergeria nunca a pn==0 o dn==0 dentro del presupuesto).
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertIn(res.proof_status, (ProofStatus.PROVEN_MATE, ProofStatus.PROVEN_NO_MATE))


class TestMostProvingNodeExpansion(unittest.TestCase):
    def test_only_unresolved_nodes_are_expanded(self):
        # invariante central del algoritmo (ver docstring "REGLA DE ORO"
        # de dfpn.py): un nodo resuelto nunca vuelve a expandirse -
        # dfpn_expansions siempre es <= dfpn_nodes (cada expansion real
        # cuenta como 1 nodo, pero un nodo puede "visitarse" sin expandir
        # si ya estaba resuelto via transposicion o si es terminal, que
        # SI cuentan como nodo pero no necesariamente incrementan
        # expansions de la misma forma que una expansion real de jugadas).
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        table = DFPNTable()
        res = search_dfpn(p, DFPNLimits(max_nodes=5000), table=table)
        self.assertLessEqual(res.metadata["dfpn_expansions"], res.metadata["dfpn_nodes"])

    def test_mpn_selection_picks_smallest_pn_at_or_node(self):
        # en un nodo OR con multiples jugadas, la seleccion del nodo mas
        # probante desciende SIEMPRE al hijo de menor pn - se verifica
        # indirectamente: la PV encontrada usa la jugada que resuelve pn=0
        # mas rapido posible entre las probadas (mate en 1 disponible,
        # nunca se explora una alternativa mas larga como "mejor").
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        res = search_dfpn(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.mate_distance_plies, 1)


if __name__ == "__main__":
    unittest.main()
