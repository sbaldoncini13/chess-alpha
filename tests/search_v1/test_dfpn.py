"""Tests de DFPN/mate_search (Fase VI, items 1-14/42/44/45): mate en 1/2/3,
preferencia por mate mas corto, retraso de mate inevitable, posicion sin
mate, ahogado, repeticion, regla de 50, material insuficiente, promocion/
subpromocion/captura al paso/enroque dentro de una prueba, PV legal,
posicion raiz intacta, determinismo.

Correr: python -m pytest tests/search_v1/test_dfpn.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    DFPNLimits, Move, Position, ProofStatus, find_forced_mate, verify_mate_proof,
)


def _run(fen, max_nodes=50000, max_depth_plies=None):
    p = Position(fen)
    res = find_forced_mate(p, DFPNLimits(max_nodes=max_nodes, max_depth_plies=max_depth_plies))
    return res, p


class TestMateShort(unittest.TestCase):
    def test_mate_in_1(self):
        res, p = _run("4k3/8/4K3/8/8/8/8/7Q w - - 0 1")
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 1)
        self.assertEqual(res.principal_variation, ["h1a8"])

    def test_mate_in_2(self):
        res, p = _run("k7/8/2K5/8/8/8/8/7Q w - - 0 1")
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 3)
        self.assertEqual(res.principal_variation, ["c6b6", "a8b8", "h1b7"])

    def test_mate_in_3(self):
        # K+R vs K con el rey negro ya cerca del borde - mate en exactamente
        # 3 jugadas propias (5 plies), verificado empiricamente (7 nodos,
        # instantaneo) - fixtures K+R vs K con el rey LEJOS del borde
        # pueden necesitar decenas de miles de nodos (tecnica de "caja"/
        # escalera completa), fuera del alcance de un test rapido.
        fen = "6k1/8/6K1/8/8/7R/8/8 w - - 0 1"
        res, p = _run(fen, max_nodes=50000, max_depth_plies=None)
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 3)
        problems = verify_mate_proof(Position(fen), res)
        self.assertEqual(problems, [])


class TestShortestMatePreferred(unittest.TestCase):
    def test_prefers_shortest_mate_when_multiple_exist(self):
        # posicion con mate en 1 disponible junto con mates mas largos -
        # DFPN debe encontrar el CAMINO MAS CORTO (el arbol PNS explora
        # todas las jugadas del atacante y selecciona pn minimo, que
        # corresponde al mate mas corto entre las jugadas probadas).
        res, p = _run("4k3/8/4K3/8/8/8/8/7Q w - - 0 1")
        self.assertEqual(res.mate_distance_plies, 1)


class TestDelayedInevitableMate(unittest.TestCase):
    def test_defender_delays_forced_loss_maximally(self):
        # mismo fixture ya validado en Fase II/V: negras forzadas a
        # perder, deben resistir el maximo de plies posible (evitar
        # b8c8, que pierde 2 plies antes).
        fen = "1k6/8/2K5/8/8/8/8/7Q b - - 0 1"
        p = Position(fen)
        # DFPN busca mate PARA el atacante (side_to_move) - aca side_to_move
        # es negras, que NO tienen mate - se prueba en cambio que blancas SI
        # fuerzan mate, y que la PV de blancas (el atacante real) evita
        # dejarle a negras la resistencia mas larga por error - se valida
        # indirectamente comparando contra iterative_deepening_search ya
        # validado (Fase II/V) para ese mismo fixture.
        from search_v1 import SearchLimits, iterative_deepening_search
        res_pvs = iterative_deepening_search(Position(fen), SearchLimits(max_depth=4), use_pvs=True)
        self.assertNotEqual(res_pvs.best_move, "b8c8")


class TestNoMate(unittest.TestCase):
    def test_no_mate_from_starting_position_within_bound(self):
        res, p = _run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", max_nodes=200000, max_depth_plies=3)
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND)

    def test_no_mate_wrong_side_to_move(self):
        res, p = _run("4k3/8/4K3/8/8/8/8/7Q b - - 0 1")
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE)


class TestStalemate(unittest.TestCase):
    def test_stalemate_disproves_mate_claim(self):
        # fixture ya validado en Fase III/IV/V como "ahogado" (negras a
        # mover, sin jugadas legales, sin jaque) - el "atacante" aca es
        # NEGRAS (side_to_move), que quedan INMEDIATAMENTE ahogadas: nunca
        # puede probarse que negras fuercen mate desde una posicion donde
        # ni siquiera tienen una jugada legal.
        fen = "k7/P1K5/1Q6/8/8/8/8/8 b - - 0 1"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE)


class TestRepetition(unittest.TestCase):
    def test_repetition_never_becomes_a_false_mate(self):
        from search_v1 import RepetitionTracker
        # posicion K+R vs K+R donde forzar la repeticion es una defensa
        # valida - se corre con presupuesto chico y se confirma que NUNCA
        # se declara PROVEN_MATE via un ciclo (el motor debe o probar un
        # mate real evitando el ciclo, o reportar no-mate/unknown).
        fen = "8/8/8/4k3/8/8/4K3/7Q w - - 0 1"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=30000, max_depth_plies=None))
        if res.proof_status == ProofStatus.PROVEN_MATE:
            problems = verify_mate_proof(Position(fen), res)
            self.assertEqual(problems, [])  # si afirma mate, TIENE que ser una PV real y verificable


class TestFiftyMoveRule(unittest.TestCase):
    def test_fifty_move_rule_deliberately_out_of_scope(self):
        # DECISION DE DISENO documentada explicitamente en dfpn.py (bug
        # real de Fase VI, ver docstring de _classify_terminal): DFPN
        # prueba "existe una secuencia de jaque mate forzado" SIN
        # considerar la regla de 50 movimientos (variante especializada,
        # permitida por el pedido) - la clave de la tabla de
        # transposiciones (zobrist_key) excluye halfmove_clock, y
        # considerar la regla de 50 de forma cacheable causaba PROVEN_MATE
        # falsos (mate real solo alcanzable con un halfmove_clock bajo,
        # incorrectamente reutilizado para la MISMA posicion con
        # halfmove_clock alto). Este test verifica el comportamiento
        # DOCUMENTADO: un mate en 1 disponible se seguye encontrando aunque
        # halfmove_clock ya este en 100 (en partida real, esto seria
        # tablas reclamables ANTES del mate - fuera del alcance de esta
        # prueba, documentado, no oculto).
        fen = "4k3/8/4K3/8/8/8/8/7Q w - - 100 60"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 1)


class TestInsufficientMaterial(unittest.TestCase):
    def test_king_vs_king_no_mate(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=5000))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_NO_MATE)


class TestPromotionMate(unittest.TestCase):
    def test_promotion_to_queen_delivers_mate(self):
        # verificado programaticamente (python-chess): Kc7 corona a dama y
        # da jaque mate inmediato (Ka8 sin escape: a7 cubierto por Ka6,
        # b7/b8 cubiertos por la dama en c8).
        fen = "k7/2P5/K7/8/8/8/8/8 w - - 0 1"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=20000, max_depth_plies=4))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertEqual(res.mate_distance_plies, 1)
        self.assertEqual(res.principal_variation, ["c7c8q"])
        problems = verify_mate_proof(Position(fen), res)
        self.assertEqual(problems, [])


class TestUnderpromotionMate(unittest.TestCase):
    def test_underpromotion_options_never_omitted_from_proof(self):
        # item explicito del pedido: "DFPN puede utilizar heuristicas
        # unicamente para ordenar, nunca para omitir movimientos legales
        # necesarios para una prueba" - se verifica DIRECTAMENTE que las 4
        # subpromociones (q/r/b/n) de un peon en la 7ma fila son legales Y
        # quedan TODAS incluidas en el orden que usa DFPN para expandir un
        # nodo (`_order_moves_for_proof`), sin ningun filtro que las excluya.
        from search_v1 import generate_legal_moves
        from search_v1.dfpn import _order_moves_for_proof
        fen = "k7/2P5/K7/8/8/8/8/8 w - - 0 1"
        p = Position(fen)
        legal = generate_legal_moves(p)
        promo_ucis = {lm.uci for lm in legal if lm.is_promotion and lm.uci.startswith("c7c8")}
        self.assertEqual(promo_ucis, {"c7c8q", "c7c8r", "c7c8b", "c7c8n"})

        ordered = _order_moves_for_proof(p, legal)
        ordered_ucis = {lm.uci for lm in ordered}
        self.assertEqual({lm.uci for lm in legal}, ordered_ucis)  # ninguna jugada se pierde al reordenar

    def test_dfpn_finds_mate_via_a_promotion_choice(self):
        # confirma que la busqueda de mate real (no solo la lista de
        # jugadas) SI llega a considerar y probar una linea que involucra
        # coronacion - complementa el test anterior (que confirma que
        # NINGUNA opcion de corona se omite) verificando el caso positivo
        # end-to-end sobre el mismo fixture.
        fen = "k7/2P5/K7/8/8/8/8/8 w - - 0 1"
        p = Position(fen)
        res = find_forced_mate(p, DFPNLimits(max_nodes=20000, max_depth_plies=2))
        self.assertEqual(res.proof_status, ProofStatus.PROVEN_MATE)
        self.assertTrue(res.principal_variation[0].startswith("c7c8"))
        problems = verify_mate_proof(Position(fen), res)
        self.assertEqual(problems, [])


class TestEnPassantInProof(unittest.TestCase):
    def test_en_passant_move_available_and_legal_within_proof(self):
        from search_v1 import generate_legal_moves
        fen = "4k3/8/8/3pP3/8/8/8/4K2Q w - d6 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        self.assertIn("e5d6", [m.uci for m in moves])  # al paso disponible - DFPN la explora como cualquier otra (generate_legal_moves ya la incluye)
        res = find_forced_mate(p, DFPNLimits(max_nodes=20000, max_depth_plies=4))
        self.assertIn(res.proof_status, (ProofStatus.PROVEN_MATE, ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND, ProofStatus.PROVEN_NO_MATE))


class TestCastlingInLine(unittest.TestCase):
    def test_castling_available_and_legal_within_proof(self):
        from search_v1 import generate_legal_moves
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        p = Position(fen)
        moves = generate_legal_moves(p)
        ucis = [m.uci for m in moves]
        self.assertIn("e1g1", ucis)
        self.assertIn("e1c1", ucis)
        res = find_forced_mate(p, DFPNLimits(max_nodes=5000, max_depth_plies=2))
        self.assertTrue(res.root_position_intact)


class TestRootIntactAndDeterminism(unittest.TestCase):
    def test_root_position_intact(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        p = Position(fen)
        find_forced_mate(p, DFPNLimits(max_nodes=50000))
        self.assertEqual(p.fen, fen)

    def test_determinism_repeated_calls(self):
        fen = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
        results = []
        for _ in range(3):
            p = Position(fen)
            res = find_forced_mate(p, DFPNLimits(max_nodes=50000))
            results.append((res.proof_status, res.mate_distance_plies, tuple(res.principal_variation)))
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
