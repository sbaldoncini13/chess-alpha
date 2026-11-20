"""Tests del SearchObserver (PASO 0, Search Tree Observatory) - ver
search_v1/observer.py y reports/search_tree_v1/OBSERVATORY_AUDIT_MATRIX.md.

Cubre exactamente lo pedido en la seccion 26 del prompt PASO 0, en lo que
aplica a esta instrumentacion (no hay tests de "roundtrip de trace en
disco" aca - eso vive en test_observatory_trace.py, junto al schema):

- tracing OFF (observer=None, default) no cambia best_move/score/nodes.
- tracing ON no cambia best_move/score (respecto de tracing OFF).
- consistencia node_id/parent_id (arbol bien formado).
- alpha/beta coherentes con el invariante fail-soft real del codigo.
- node_class (PV/CUT/ALL) coherente con alpha_in/beta_in/returned_score.
- exit_recorded=True en TODOS los eventos (ninguna excepcion dejo un nodo
  a medio registrar).
- iteraciones de iterative deepening: una entrada por profundidad
  completada, nodes_cumulative monotono creciente.
- roundtrip serialize/deserialize (JSON) del propio observer.to_dict().
- max_events trunca el detalle sin romper la busqueda.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json

from search_v1 import (
    INFINITY,
    OrderingState,
    Position,
    QuiescenceLimits,
    RepetitionTracker,
    SearchContext,
    SearchLimits,
    SearchObserver,
    TranspositionTable,
    iterative_deepening_search,
    pvs,
    quiescence,
)

_OPENING_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_MATE_IN_3_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
_MIDDLEGAME_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
_ENDGAME_FEN = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"
_TACTICAL_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

_SAMPLE_FENS = [_OPENING_FEN, _MATE_IN_3_FEN, _MIDDLEGAME_FEN, _ENDGAME_FEN, _TACTICAL_FEN]


def _run(fen, depth=3, observer=None, max_qnodes=8000):
    position = Position(fen)
    limits = SearchLimits(max_depth=depth)
    qlimits = QuiescenceLimits(max_qdepth=4, max_qnodes=max_qnodes)
    result = iterative_deepening_search(
        position, limits, algorithm="alpha_beta", use_pvs=True,
        tt=TranspositionTable(4096), ordering_state=OrderingState(),
        quiescence_limits=qlimits, observer=observer,
    )
    return result, position


class TestObserverDoesNotChangeBehavior(unittest.TestCase):
    """Requisito de seguridad central (item 2 del pedido): CERO cambio de
    comportamiento con observer=None, y CERO cambio de RESULTADO con
    observer activo (solo se agrega telemetria de solo lectura)."""

    def test_default_observer_is_none(self):
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        self.assertIsNone(ctx.observer)

    def test_same_result_with_and_without_observer_across_sample(self):
        for fen in _SAMPLE_FENS:
            with self.subTest(fen=fen):
                r_off, pos_off = _run(fen, observer=None)
                r_on, pos_on = _run(fen, observer=SearchObserver())
                self.assertEqual(r_off.best_move, r_on.best_move)
                self.assertEqual(r_off.score, r_on.score)
                self.assertEqual(r_off.nodes, r_on.nodes)
                self.assertEqual(r_off.completed_depth, r_on.completed_depth)
                self.assertEqual(r_off.principal_variation, r_on.principal_variation)
                # la posicion nunca queda mutada, con o sin observer (invariante ya existente, Fase II)
                self.assertEqual(pos_off.fen, Position(fen).fen)
                self.assertEqual(pos_on.fen, Position(fen).fen)

    def test_pvs_direct_call_identical_with_and_without_observer(self):
        p1 = Position(_MIDDLEGAME_FEN)
        ctx1 = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        score1, pv1 = pvs(p1, 3, -INFINITY, INFINITY, ctx1)

        p2 = Position(_MIDDLEGAME_FEN)
        ctx2 = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        ctx2.observer = SearchObserver()
        score2, pv2 = pvs(p2, 3, -INFINITY, INFINITY, ctx2)

        self.assertEqual(score1, score2)
        self.assertEqual(pv1, pv2)
        self.assertEqual(ctx1.nodes, ctx2.nodes)

    def test_quiescence_direct_call_identical_with_and_without_observer(self):
        qlimits = QuiescenceLimits(max_qdepth=4, max_qnodes=2000)

        p1 = Position(_TACTICAL_FEN)
        ctx1 = SearchContext(SearchLimits(max_depth=2), RepetitionTracker(), quiescence_limits=qlimits)
        score1, pv1 = quiescence(p1, -INFINITY, INFINITY, ctx1, qlimits, qlimits.max_qdepth, qlimits.max_check_extensions)

        p2 = Position(_TACTICAL_FEN)
        ctx2 = SearchContext(SearchLimits(max_depth=2), RepetitionTracker(), quiescence_limits=qlimits)
        ctx2.observer = SearchObserver()
        score2, pv2 = quiescence(p2, -INFINITY, INFINITY, ctx2, qlimits, qlimits.max_qdepth, qlimits.max_check_extensions)

        self.assertEqual(score1, score2)
        self.assertEqual(pv1, pv2)
        self.assertEqual(ctx1.q_nodes, ctx2.q_nodes)


class TestObserverTraceIntegrity(unittest.TestCase):
    """Consistencia estructural del trace en si (item 26: node parent/
    child consistency, alpha/beta events coherentes, qnodes consistentes)."""

    def test_all_events_have_exit_recorded(self):
        _, _ = _run(_MIDDLEGAME_FEN, observer=(obs := SearchObserver()))
        self.assertGreater(len(obs.events), 0)
        self.assertTrue(all(e.exit_recorded for e in obs.events))

    def test_parent_child_consistency(self):
        _, _ = _run(_TACTICAL_FEN, observer=(obs := SearchObserver()))
        for event in obs.events:
            if event.parent_id is not None:
                self.assertGreaterEqual(event.parent_id, 0)
                self.assertLess(event.parent_id, event.node_id, "parent_id debe preceder a node_id")

    def test_node_ids_are_unique_and_sequential(self):
        _, _ = _run(_MIDDLEGAME_FEN, observer=(obs := SearchObserver()))
        ids = [e.node_id for e in obs.events]
        self.assertEqual(ids, list(range(len(ids))))

    def test_alpha_beta_fail_soft_invariant(self):
        """beta_out == beta_in siempre (nunca se reasigna dentro de un
        nodo); alpha_out >= alpha_in siempre (fail-soft, alpha solo
        aumenta) - ver derivacion documentada en observer.py::_classify."""
        _, _ = _run(_TACTICAL_FEN, depth=3, observer=(obs := SearchObserver()))
        self.assertGreater(len(obs.events), 0)
        for event in obs.events:
            self.assertEqual(event.beta_out, event.beta_in)
            self.assertGreaterEqual(event.alpha_out, event.alpha_in)

    def test_node_class_matches_alpha_beta_score(self):
        """PV/CUT/ALL derivado (event.node_class) debe ser consistente con
        la definicion aplicada a alpha_in/beta_in/returned_score - re-
        deriva independientemente y compara, no confia en el campo solo."""
        _, _ = _run(_MIDDLEGAME_FEN, depth=3, observer=(obs := SearchObserver()))
        for event in obs.events:
            if event.returned_score >= event.beta_in:
                expected = "CUT"
            elif event.returned_score <= event.alpha_in:
                expected = "ALL"
            else:
                expected = "PV"
            self.assertEqual(event.node_class, expected)

    def test_qnodes_match_context_counter(self):
        result, _ = _run(_TACTICAL_FEN, depth=2, observer=(obs := SearchObserver()))
        n_quiescence_events = sum(1 for e in obs.events if e.node_type == "quiescence")
        # el observer cuenta TODAS las llamadas a quiescence() (incluidas
        # las que abortan por max_check_extensions antes de llegar a
        # context.q_nodes+=1... no, visit_qnode() es lo PRIMERO que hace
        # _quiescence_impl, antes de cualquier chequeo - deben coincidir 1:1.
        self.assertEqual(n_quiescence_events, result.metadata["q_nodes"])

    def test_search_nodes_match_context_counter(self):
        result, _ = _run(_MIDDLEGAME_FEN, depth=3, observer=(obs := SearchObserver()))
        n_search_events = sum(1 for e in obs.events if e.node_type == "search")
        self.assertEqual(n_search_events, result.nodes)

    def test_cutoff_move_is_from_the_recorded_search_order(self):
        _, _ = _run(_MIDDLEGAME_FEN, depth=3, observer=(obs := SearchObserver()))
        checked_any = False
        for event in obs.events:
            if event.cutoff_move_uci is not None and event.search_order is not None:
                checked_any = True
                self.assertIn(event.cutoff_move_uci, event.search_order)
                self.assertEqual(event.search_order[event.cutoff_move_index], event.cutoff_move_uci)
        self.assertTrue(checked_any, "la muestra deberia producir al menos un cutoff con datos completos")


class TestObserverIterations(unittest.TestCase):
    """Seccion 11 del pedido (Iterative Deepening) - una entrada por
    profundidad completada, monotonamente creciente en nodes/tiempo."""

    def test_one_entry_per_completed_depth(self):
        result, _ = _run(_MIDDLEGAME_FEN, depth=4, observer=(obs := SearchObserver()))
        self.assertEqual(len(obs.iterations), result.completed_depth)
        self.assertEqual([it["depth"] for it in obs.iterations], list(range(1, result.completed_depth + 1)))

    def test_nodes_cumulative_monotonic(self):
        _, _ = _run(_TACTICAL_FEN, depth=4, observer=(obs := SearchObserver()))
        nodes_seq = [it["nodes_cumulative"] for it in obs.iterations]
        self.assertEqual(nodes_seq, sorted(nodes_seq))
        if len(nodes_seq) > 1:
            self.assertLess(nodes_seq[0], nodes_seq[-1])


class TestObserverSerialization(unittest.TestCase):
    def test_to_dict_json_roundtrip(self):
        _, _ = _run(_MIDDLEGAME_FEN, observer=(obs := SearchObserver()))
        payload = obs.to_dict()
        serialized = json.dumps(payload)
        restored = json.loads(serialized)
        self.assertEqual(restored["n_events_recorded"], len(obs.events))
        self.assertEqual(len(restored["events"]), len(obs.events))
        self.assertEqual(len(restored["iterations"]), len(obs.iterations))


class TestObserverMaxEvents(unittest.TestCase):
    """Control de tamano del trace (item 7 del pedido) - un tope no debe
    romper la busqueda ni el resultado, solo el DETALLE guardado."""

    def test_max_events_truncates_without_changing_result(self):
        r_full, _ = _run(_MIDDLEGAME_FEN, depth=3, observer=SearchObserver())
        r_capped, _ = _run(_MIDDLEGAME_FEN, depth=3, observer=(obs_capped := SearchObserver(max_events=50)))
        self.assertEqual(r_full.best_move, r_capped.best_move)
        self.assertEqual(r_full.score, r_capped.score)
        self.assertEqual(r_full.nodes, r_capped.nodes)
        self.assertLessEqual(len(obs_capped.events), 50)
        self.assertGreater(obs_capped._truncated_count, 0)


if __name__ == "__main__":
    unittest.main()
