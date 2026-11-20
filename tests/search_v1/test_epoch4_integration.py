"""Tests de integracion Epoch4 ordering-only provider (Fase VII).

Tier 1 (unitario, rapido, evaluador STUB salvo TestEpoch4WithRealEvaluator):
cache direct-mapped, correccion de reorder_with_epoch4 (permutacion exacta,
fallback de vocabulario), desactivado por defecto, aislamiento de latencia
(regresion contra el bug real ~2.2x del motor viejo por doble inferencia
via top_k_moves()).

Tier 2 (regresion posicion-por-posicion, TOLERANCIA CERO, no promediada -
misma metodologia "anti-promedio" de Fase V/VI): TestEpoch4NeverChangesBestMove
corre sobre una muestra chica fija por defecto - el gate REAL sobre el
corpus completo (39.961 + 12.963 Nivel C + 2 historicos) vive en
src/151_phase7_dataset_validation.py por costo (invariante #13 del
proyecto: ninguna corrida larga se lanza sin estimacion de tiempo previa).

Correr: python -m pytest tests/search_v1/test_epoch4_integration.py -v
"""

import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from search_v1 import (
    INFINITY,
    DFPNTable,
    Epoch4OrderingCache,
    Epoch4OrderingConfig,
    Position,
    RepetitionTracker,
    SearchContext,
    SearchLimits,
    TranspositionTable,
    generate_legal_moves,
    iterative_deepening_search,
    pvs,
)
from search_v1.epoch4_provider import evaluate_epoch4, reorder_with_epoch4

_OPENING_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_MATE_IN_3_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
_MIDDLEGAME_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

# Muestra chica y fija de FENs para regresion posicion-por-posicion en
# pytest (Tier 2 "smoke") - el corpus completo (dataset + Nivel C +
# historicos) es responsabilidad de src/151_phase7_dataset_validation.py,
# no de la suite de pytest por defecto.
_TIER2_SMOKE_FENS = [
    _OPENING_FEN,
    _MATE_IN_3_FEN,
    _MIDDLEGAME_FEN,
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",  # final de peon simple
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # posicion tactica clasica (Kiwipete)
]


class _StubEvaluator:
    """Evaluador stub - `.predict()` devuelve probabilidades fijas/
    controladas sin cargar ningun modelo real. Cuenta llamadas a
    `.predict()`/`.top_k_moves()` para verificar inferencia unica (Tier 1)
    y para detectar si algo llama por error a `.top_k_moves()` (la causa
    real, ya documentada en epoch4_provider.py, de la regresion de
    latencia ~2.2x del motor viejo)."""

    def __init__(self, probs_by_fen=None, sleep_seconds: float = 0.0):
        self.probs_by_fen = probs_by_fen or {}
        self.sleep_seconds = sleep_seconds
        self.predict_calls = 0
        self.top_k_moves_calls = 0

    def predict(self, board):
        self.predict_calls += 1
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        probs = self.probs_by_fen.get(board.fen(), {})
        return dict(probs), 0.5

    def top_k_moves(self, board, k: int = 8):
        self.top_k_moves_calls += 1
        return []


class TestEpoch4OrderingCache(unittest.TestCase):
    def test_store_then_probe_hit(self):
        cache = Epoch4OrderingCache()
        cache.store(12345, {"e2e4": 0.8})
        probs = cache.probe(12345)
        self.assertEqual(probs, {"e2e4": 0.8})
        self.assertEqual(cache.hits, 1)

    def test_index_collision_detected_full_key_prevents_false_hit(self):
        # capacidad=1 fuerza a que CUALQUIER clave distinta comparta el
        # mismo indice - mismo patron ya validado para TranspositionTable/
        # DFPNTable: la clave COMPLETA debe evitar el falso hit.
        cache = Epoch4OrderingCache(capacity_entries=1)
        cache.store(111, {"e2e4": 1.0})
        probs = cache.probe(222)  # mismo indice (0), clave distinta
        self.assertIsNone(probs)
        self.assertEqual(cache.collisions, 1)

    def test_miss_on_empty_cache(self):
        cache = Epoch4OrderingCache()
        self.assertIsNone(cache.probe(999))
        self.assertEqual(cache.probes, 1)
        self.assertEqual(cache.hits, 0)

    def test_namespace_separate_from_alpha_beta_tt_and_dfpn_table(self):
        self.assertFalse(issubclass(Epoch4OrderingCache, TranspositionTable))
        self.assertFalse(issubclass(Epoch4OrderingCache, DFPNTable))
        # instanciar los 3 juntos y confirmar que no comparten estado
        tt = TranspositionTable(64)
        dfpn_table = DFPNTable(64)
        epoch4_cache = Epoch4OrderingCache(64)
        epoch4_cache.store(1, {"e2e4": 1.0})
        self.assertIsNone(tt.probe(1))
        self.assertIsNone(dfpn_table.probe(1, "white", None))

    def test_evaluate_epoch4_calls_predict_exactly_once_per_position(self):
        p = Position(_OPENING_FEN)
        evaluator = _StubEvaluator(probs_by_fen={_OPENING_FEN: {"g8f6": 0.9}})
        cache = Epoch4OrderingCache()

        probs1 = evaluate_epoch4(p, evaluator, cache)
        probs2 = evaluate_epoch4(p, evaluator, cache)  # misma posicion - debe ser cache hit

        self.assertEqual(probs1, {"g8f6": 0.9})
        self.assertEqual(probs2, {"g8f6": 0.9})
        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(cache.hits, 1)

    def test_evaluate_epoch4_without_cache_calls_predict_every_time(self):
        p = Position(_OPENING_FEN)
        evaluator = _StubEvaluator(probs_by_fen={_OPENING_FEN: {"g8f6": 0.9}})

        evaluate_epoch4(p, evaluator, cache=None)
        evaluate_epoch4(p, evaluator, cache=None)

        self.assertEqual(evaluator.predict_calls, 2)


class TestEpoch4ReorderCorrectness(unittest.TestCase):
    def setUp(self):
        self.moves = generate_legal_moves(Position(_MIDDLEGAME_FEN))
        self.assertGreater(len(self.moves), 5)

    def test_output_is_exact_permutation_of_input(self):
        rng = random.Random(20260809)
        for _ in range(20):
            probs = {lm.uci: rng.random() for lm in rng.sample(self.moves, k=rng.randint(0, len(self.moves)))}
            result = reorder_with_epoch4(self.moves, hash_move_uci=None, probs=probs)
            self.assertEqual(len(result), len(self.moves))
            self.assertEqual({lm.uci for lm in result}, {lm.uci for lm in self.moves})

    def test_hash_move_stays_pinned_at_index_zero(self):
        hash_move = self.moves[0].uci
        # probs favorece fuertemente OTRA jugada - el hash move debe seguir
        # fijo en indice 0 de todas formas.
        other = next(lm.uci for lm in self.moves if lm.uci != hash_move)
        probs = {other: 0.99, hash_move: 0.01}
        result = reorder_with_epoch4(self.moves, hash_move_uci=hash_move, probs=probs)
        self.assertEqual(result[0].uci, hash_move)

    def test_moves_outside_vocab_keep_classical_relative_order(self):
        # ningun uci esta en probs (vocabulario "vacio") - fallback total:
        # el orden clasico de entrada debe quedar intacto.
        result = reorder_with_epoch4(self.moves, hash_move_uci=None, probs={})
        self.assertEqual([lm.uci for lm in result], [lm.uci for lm in self.moves])

    def test_covered_moves_sorted_by_probability_descending(self):
        a, b, c = self.moves[0].uci, self.moves[1].uci, self.moves[2].uci
        probs = {a: 0.1, b: 0.9, c: 0.5}
        result = reorder_with_epoch4(self.moves, hash_move_uci=None, probs=probs)
        covered_order = [lm.uci for lm in result if lm.uci in probs]
        self.assertEqual(covered_order, [b, c, a])

    def test_partial_coverage_keeps_uncovered_moves_in_classical_order(self):
        covered_uci = self.moves[-1].uci  # deliberadamente la ULTIMA jugada clasica
        probs = {covered_uci: 0.99}
        result = reorder_with_epoch4(self.moves, hash_move_uci=None, probs=probs)
        # la jugada cubierta pasa al frente (unica en `probs`); el resto
        # conserva el orden clasico relativo original.
        self.assertEqual(result[0].uci, covered_uci)
        expected_uncovered = [lm.uci for lm in self.moves if lm.uci != covered_uci]
        self.assertEqual([lm.uci for lm in result[1:]], expected_uncovered)

    def test_empty_move_list_returns_empty(self):
        self.assertEqual(reorder_with_epoch4([], hash_move_uci=None, probs={}), [])


class TestEpoch4DisabledByDefault(unittest.TestCase):
    def test_config_default_is_disabled(self):
        self.assertFalse(Epoch4OrderingConfig().enabled)

    def test_pvs_with_no_epoch4_kwargs_matches_pvs_with_disabled_config(self):
        p1 = Position(_MIDDLEGAME_FEN)
        ctx1 = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        score1, pv1 = pvs(p1, 2, -INFINITY, INFINITY, ctx1)

        p2 = Position(_MIDDLEGAME_FEN)
        ctx2 = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())
        score2, pv2 = pvs(
            p2, 2, -INFINITY, INFINITY, ctx2,
            epoch4_config=Epoch4OrderingConfig(enabled=False), epoch4_evaluator=_StubEvaluator(),
        )

        self.assertEqual(score1, score2)
        self.assertEqual(pv1, pv2)
        self.assertEqual(ctx2.epoch4_cache_probes, 0)
        self.assertEqual(ctx2.epoch4_inferences, 0)
        self.assertEqual(ctx2.epoch4_positions_reordered, 0)

    def test_iterative_deepening_rejects_epoch4_config_under_negamax_mode(self):
        p = Position(_OPENING_FEN)
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                p, SearchLimits(max_depth=1), algorithm="alpha_beta", use_pvs=False,
                epoch4_config=Epoch4OrderingConfig(enabled=True), epoch4_evaluator=_StubEvaluator(),
            )

    def test_iterative_deepening_rejects_epoch4_config_under_plain_negamax(self):
        p = Position(_OPENING_FEN)
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                p, SearchLimits(max_depth=1), algorithm="negamax",
                epoch4_config=Epoch4OrderingConfig(enabled=True),
            )


class TestEpoch4Scope(unittest.TestCase):
    """Fase VIII - epoch4_config.scope, agregado ADITIVO tras medir el
    costo real de Epoch4 por-nodo en el micro-benchmark A/B/C/D.
    "all_nodes" es y sigue siendo el default (comportamiento original de
    Fase VII, sin cambios) - "root_only" es opt-in nuevo."""

    def test_default_scope_is_all_nodes(self):
        self.assertEqual(Epoch4OrderingConfig().scope, "all_nodes")

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            Epoch4OrderingConfig(scope="only_at_night")

    def test_all_nodes_scope_evaluates_at_every_node(self):
        evaluator = _StubEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}})
        p = Position(_MIDDLEGAME_FEN)
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        pvs(
            p, 3, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True, scope="all_nodes"),
        )
        self.assertGreater(evaluator.predict_calls, 1)  # se evaluo en mas de un nodo

    def test_root_only_scope_evaluates_only_at_root(self):
        evaluator = _StubEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}})
        p = Position(_MIDDLEGAME_FEN)
        ctx = SearchContext(SearchLimits(max_depth=3), RepetitionTracker())
        pvs(
            p, 3, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True, scope="root_only"),
        )
        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(ctx.epoch4_positions_reordered, 1)


class TestEpoch4LatencyIsolation(unittest.TestCase):
    def test_inference_time_scales_with_predict_calls_not_cache_hits(self):
        sleep_s = 0.02
        evaluator = _StubEvaluator(probs_by_fen={_OPENING_FEN: {}}, sleep_seconds=sleep_s)
        cache = Epoch4OrderingCache()
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        p = Position(_OPENING_FEN)

        evaluate_epoch4(p, evaluator, cache, ctx)  # miss - paga la latencia
        for _ in range(5):
            evaluate_epoch4(p, evaluator, cache, ctx)  # 5 hits - no deberian sumar latencia

        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(ctx.epoch4_cache_hits, 5)
        # tolerancia generosa (I/O de test compartido) pero acotada: 5 hits
        # gratis no deben acercarse a 5*sleep_s adicionales.
        self.assertLess(ctx.epoch4_inference_time_ms, sleep_s * 1000 * 3)
        self.assertGreaterEqual(ctx.epoch4_inference_time_ms, sleep_s * 1000 * 0.5)

    def test_top_k_moves_is_never_called_by_epoch4_provider(self):
        evaluator = _StubEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}})
        p = Position(_MIDDLEGAME_FEN)
        cache = Epoch4OrderingCache()
        ctx = SearchContext(SearchLimits(max_depth=2), RepetitionTracker())

        pvs(
            p, 2, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True), epoch4_cache=cache,
        )

        self.assertEqual(evaluator.top_k_moves_calls, 0)
        self.assertGreater(evaluator.predict_calls, 0)


class TestEpoch4NeverChangesTacticalTruth(unittest.TestCase):
    """Tier 2 - regresion posicion-por-posicion, TOLERANCIA CERO sobre el
    invariante REAL y matematicamente garantizado: el SCORE final de PVS
    (el "valor minimax", la verdad tactica) es order-invariant, EXACTAMENTE
    el mismo criterio ya usado por Fase V para validar PVS vs Alpha-Beta
    (ver test_pvs.py::TestPVSMatchesAlphaBeta - compara score, nunca
    best_move literal).

    HALLAZGO DOCUMENTADO durante la escritura de este test (invariante #4
    del proyecto, "no asumir" - se investigo antes de "corregir" el test):
    una primera version de este test comparaba `best_move` literal y
    fallaba en varias posiciones. Investigado: el score SIEMPRE coincidia
    (confirmado con los 2 historicos de defensa obligatoria, ver abajo) -
    la diferencia de `best_move` ocurre SOLO cuando dos o mas jugadas
    empatan en score exacto (comun a profundidad chica con evaluacion
    material pura), y "cual de las empatadas gana" es legitimamente
    dependiente del orden - el mismo fenomeno que la propia Fase V ya
    documenta y por el que su propio gate compara SOLO score. Exigir
    igualdad de `best_move` bajo empate seria un criterio MAS ESTRICTO que
    el que el proyecto ya valido para PVS mismo, no una regresion real de
    Fase VII. La igualdad de SCORE es la verdad tactica; la eleccion entre
    jugadas objetivamente equivalentes no lo es.

    El gate REAL sobre el corpus completo (incluyendo la comparacion
    dedicada de `defensa_obligatoria` con `safe_move_count==1`, donde por
    construccion NO hay empate entre la jugada segura y las perdedoras)
    vive en src/151_phase7_dataset_validation.py (costo no trivial,
    requiere estimacion de tiempo previa por invariante #13)."""

    def test_score_identical_with_and_without_epoch4_smoke_sample(self):
        rng = random.Random(20260809)
        for fen in _TIER2_SMOKE_FENS:
            with self.subTest(fen=fen):
                p_without = Position(fen)
                result_without = iterative_deepening_search(
                    p_without, SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                )

                legal = generate_legal_moves(Position(fen))
                fake_probs = {lm.uci: rng.random() for lm in legal}
                evaluator = _StubEvaluator(probs_by_fen={fen: fake_probs})
                p_with = Position(fen)
                result_with = iterative_deepening_search(
                    p_with, SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True),
                )

                self.assertEqual(result_with.score, result_without.score)

    def test_historical_forced_defense_score_unaffected_by_adversarial_epoch4(self):
        # Los 2 historicos de tests/regression/forced_defense_critical.json
        # (Fase 7 del "Policy Prior" viejo, RECHAZADO por ignorar la
        # defensa correcta en estos mismos 2 casos) - probs deliberadamente
        # sesgadas EN CONTRA de la solucion documentada (`solution`), el
        # escenario adversarial exacto que causo el rechazo original.
        cases = [
            ("1kr5/1p3p1p/8/4Q3/1P1RPP2/8/PKP1q3/8 b - - 8 35", "b8a7"),
            ("5R2/4N1pk/1p3n1p/1P5P/8/r6K/5PP1/8 w - - 13 44", "f2f3"),
        ]
        for fen, solution in cases:
            with self.subTest(fen=fen):
                result_without = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                )
                self.assertEqual(result_without.best_move, solution)  # control: el baseline ya acierta a esta profundidad

                legal = generate_legal_moves(Position(fen))
                adversarial_probs = {lm.uci: (0.99 if lm.uci != solution else 0.01) for lm in legal}
                evaluator = _StubEvaluator(probs_by_fen={fen: adversarial_probs})
                result_with = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True),
                )
                # invariante real: el SCORE (verdad tactica) no empeora ni
                # cambia por la presencia adversarial de Epoch4, sin
                # importar cual jugada empatada termine eligiendose.
                self.assertEqual(result_with.score, result_without.score)


class TestEpoch4WithRealEvaluator(unittest.TestCase):
    """Subconjunto CHICO con el HybridNeuralEvaluator real (checkpoint
    Epoch4 congelado) - se salta automaticamente si el checkpoint no esta
    disponible en este entorno."""

    @classmethod
    def setUpClass(cls):
        try:
            from chess_alpha.config import LICHESS_POLICY_VOCAB_PATH, SELECTED_MODEL_PATH
            from chess_alpha.inference import HybridNeuralEvaluator
        except ImportError as exc:
            raise unittest.SkipTest(f"chess_alpha no disponible: {exc}")
        if not SELECTED_MODEL_PATH.exists():
            raise unittest.SkipTest(f"checkpoint Epoch4 no encontrado en {SELECTED_MODEL_PATH}")
        cls.evaluator = HybridNeuralEvaluator(SELECTED_MODEL_PATH, LICHESS_POLICY_VOCAB_PATH)

    def test_predict_output_is_legal_mask_correct(self):
        import chess
        for fen in _TIER2_SMOKE_FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                probs, _value = self.evaluator.predict(board)
                legal_ucis = {m.uci() for m in board.legal_moves}
                self.assertTrue(set(probs.keys()).issubset(legal_ucis))

    def test_reordering_is_genuinely_different_from_classical_for_at_least_one_fen(self):
        cache = Epoch4OrderingCache()
        any_reordered = False
        for fen in _TIER2_SMOKE_FENS:
            p = Position(fen)
            legal_moves = generate_legal_moves(p)
            probs = evaluate_epoch4(p, self.evaluator, cache)
            reordered = reorder_with_epoch4(legal_moves, hash_move_uci=None, probs=probs)
            if [lm.uci for lm in reordered] != [lm.uci for lm in legal_moves]:
                any_reordered = True
        self.assertTrue(any_reordered, "Epoch4 nunca reordeno nada en la muestra - wiring probablemente no-op")

    def test_score_identical_with_and_without_real_epoch4(self):
        # invariante real (ver TestEpoch4NeverChangesTacticalTruth para la
        # justificacion completa de por que se compara SCORE y no
        # best_move literal - empates de score bajo evaluacion material
        # pura hacen que "cual jugada empatada gana" sea legitimamente
        # dependiente del orden, igual que ya documenta Fase V).
        for fen in _TIER2_SMOKE_FENS:
            with self.subTest(fen=fen):
                result_without = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                )
                result_with = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=self.evaluator, epoch4_config=Epoch4OrderingConfig(enabled=True),
                )
                self.assertEqual(result_with.score, result_without.score)


if __name__ == "__main__":
    unittest.main()
