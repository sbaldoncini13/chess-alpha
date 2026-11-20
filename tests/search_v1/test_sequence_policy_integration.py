"""Tests de integracion Sequence Policy V1 FULL + Fusion con Epoch4 (Fase
VIII).

Tier 1 (unitario, rapido, evaluador STUB salvo Tier 1.5): cache con clave
compuesta (zobrist+historia), correccion de reorder_with_sequence_policy/
reorder_with_fusion (permutacion exacta, fallback de cobertura), scope
root-only realmente forzado (no solo declarado), desactivado por defecto,
las 4 validaciones fail-fast de iterative_deepening_search, cadena de
fallback (4 escenarios), aislamiento de latencia, regresion del retrofit
try/except de Epoch4 (Fase VII). Tambien cubre las invariantes de
`sequence_policy_inference.evaluator` pedidas explicitamente: historia
debe reproducir el board real, nunca se agrega EOS, truncado correcto de
historia larga (>255) - todo esto testeable SIN cargar el checkpoint real
(funciones puras).

Tier 1.5 (modelo real, `setUpClass` skip-si-no-disponible, mismo patron
que TestEpoch4WithRealEvaluator): SHA256 verificado como fallo duro (no
solo condicion de skip), legal-mask correcta, caso EN DISTRIBUCION real
(board despues de e2e4 + history=['e2e4'] - nunca "posicion inicial +
historia de otra posicion", el bug que casi se cuela en este test segun
la revision pedida), historia larga real, historia incompatible ->
fallback/error controlado.

Tier 2 (regresion de score, TOLERANCIA CERO, no promediada - mismo
criterio "anti-promedio" ya establecido en test_epoch4_integration.py):
modos A/B/C/D sobre la muestra smoke, con verificacion de SCORE (no
best_move literal bajo empates, ver TestEpoch4NeverChangesTacticalTruth
para la justificacion completa ya documentada).

Correr: python -m pytest tests/search_v1/test_sequence_policy_integration.py -v
"""

import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chess

from search_v1 import (
    INFINITY,
    Epoch4OrderingConfig,
    FusionConfig,
    Position,
    RepetitionTracker,
    SearchContext,
    SearchLimits,
    SequencePolicyCache,
    SequencePolicyConfig,
    generate_legal_moves,
    iterative_deepening_search,
    pvs,
)
from search_v1.candidate_fusion import reorder_with_fusion
from search_v1.sequence_policy_provider import evaluate_sequence_policy, reorder_with_sequence_policy
from sequence_policy_inference.evaluator import (
    BOS_STD_ID,
    SequencePolicyHistoryMismatchError,
    build_input_action_ids,
    history_matches_board,
    replay_history_ucis,
)

_OPENING_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_MATE_IN_3_FEN = "k7/8/2K5/8/8/8/8/7Q w - - 0 1"
_MIDDLEGAME_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

_TIER2_SMOKE_FENS = [
    _OPENING_FEN,
    _MATE_IN_3_FEN,
    _MIDDLEGAME_FEN,
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]

_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "Chess_Alpha_Sequence_Policy_FULL_2286_PC" / "last.ckpt"
)
_EXPECTED_CHECKPOINT_SHA256 = "29f96d5a5343931e4cab71c6affa7628ab97240459a6792d7314f51901644804"


class _StubSequenceEvaluator:
    """Evaluador stub de Sequence Policy - `.predict(board, history_ucis)`
    devuelve un dict fijo (sin modelo real). Cuenta llamadas para verificar
    scope root-only realmente forzado en la practica, no solo declarado."""

    def __init__(self, probs_by_fen=None, sleep_seconds: float = 0.0):
        self.probs_by_fen = probs_by_fen or {}
        self.sleep_seconds = sleep_seconds
        self.predict_calls = 0

    def predict(self, board, history_ucis):
        self.predict_calls += 1
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return dict(self.probs_by_fen.get(board.fen(), {}))


class _StubEpoch4Evaluator:
    """Evaluador stub de EPOCH4 - `.predict(board) -> (probs, value)`
    (firma de UN solo argumento + tupla, distinta de
    _StubSequenceEvaluator a proposito, para no confundir los dos roles
    en los tests de fallback/fusion)."""

    def __init__(self, probs_by_fen=None):
        self.probs_by_fen = probs_by_fen or {}
        self.predict_calls = 0

    def predict(self, board):
        self.predict_calls += 1
        return dict(self.probs_by_fen.get(board.fen(), {})), 0.5


class _RaisingEvaluator:
    """Evaluador que siempre lanza excepcion en `.predict()` - simula OOM/
    fallo de inferencia real para probar la cadena de fallback."""

    def predict(self, *args, **kwargs):
        raise RuntimeError("fallo simulado de inferencia")


# ---------------------------------------------------------------------
# Tier 1 - cache con clave compuesta
# ---------------------------------------------------------------------
class TestSequencePolicyCache(unittest.TestCase):
    def test_store_then_probe_hit(self):
        cache = SequencePolicyCache()
        key = SequencePolicyCache.compute_key(12345, ["e2e4"])
        cache.store(key, {"e7e5": 0.5})
        self.assertEqual(cache.probe(key), {"e7e5": 0.5})
        self.assertEqual(cache.hits, 1)

    def test_index_collision_detected_full_key_prevents_false_hit(self):
        cache = SequencePolicyCache(capacity_entries=1)
        key1 = SequencePolicyCache.compute_key(111, [])
        key2 = SequencePolicyCache.compute_key(222, [])
        cache.store(key1, {"e2e4": 1.0})
        self.assertIsNone(cache.probe(key2))
        self.assertEqual(cache.collisions, 1)

    def test_different_history_same_zobrist_key_caches_separately(self):
        # LA correccion especifica de Sequence Policy: dos historias
        # DISTINTAS que llegan a la MISMA posicion (mismo zobrist_key) no
        # deben compartir entrada de cache - la premisa completa del
        # modelo es que la historia importa (transposicion/plan).
        zobrist = 999
        key_a = SequencePolicyCache.compute_key(zobrist, ["e2e4", "e7e5"])
        key_b = SequencePolicyCache.compute_key(zobrist, ["g1f3", "b8c6"])
        self.assertNotEqual(key_a, key_b)

        cache = SequencePolicyCache()
        cache.store(key_a, {"g1f3": 0.9})
        cache.store(key_b, {"e2e4": 0.7})
        self.assertEqual(cache.probe(key_a), {"g1f3": 0.9})
        self.assertEqual(cache.probe(key_b), {"e2e4": 0.7})

    def test_evaluate_sequence_policy_calls_predict_exactly_once_per_key(self):
        p = Position(_OPENING_FEN)
        evaluator = _StubSequenceEvaluator(probs_by_fen={_OPENING_FEN: {"g8f6": 0.8}})
        cache = SequencePolicyCache()

        probs1 = evaluate_sequence_policy(p, evaluator, cache, [])
        probs2 = evaluate_sequence_policy(p, evaluator, cache, [])

        self.assertEqual(probs1, {"g8f6": 0.8})
        self.assertEqual(probs2, {"g8f6": 0.8})
        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(cache.hits, 1)


# ---------------------------------------------------------------------
# Tier 1 - reorder_with_sequence_policy / reorder_with_fusion (puras)
# ---------------------------------------------------------------------
class TestSequenceReorderCorrectness(unittest.TestCase):
    def setUp(self):
        self.moves = generate_legal_moves(Position(_MIDDLEGAME_FEN))
        self.assertGreater(len(self.moves), 5)

    def test_output_is_exact_permutation_of_input(self):
        rng = random.Random(20260810)
        for _ in range(20):
            probs = {lm.uci: rng.random() for lm in rng.sample(self.moves, k=rng.randint(0, len(self.moves)))}
            result = reorder_with_sequence_policy(self.moves, hash_move_uci=None, probs=probs)
            self.assertEqual(len(result), len(self.moves))
            self.assertEqual({lm.uci for lm in result}, {lm.uci for lm in self.moves})

    def test_hash_move_stays_pinned_at_index_zero(self):
        hash_move = self.moves[0].uci
        other = next(lm.uci for lm in self.moves if lm.uci != hash_move)
        probs = {other: 0.99, hash_move: 0.01}
        result = reorder_with_sequence_policy(self.moves, hash_move_uci=hash_move, probs=probs)
        self.assertEqual(result[0].uci, hash_move)

    def test_moves_outside_coverage_keep_classical_relative_order(self):
        result = reorder_with_sequence_policy(self.moves, hash_move_uci=None, probs={})
        self.assertEqual([lm.uci for lm in result], [lm.uci for lm in self.moves])

    def test_empty_move_list_returns_empty(self):
        self.assertEqual(reorder_with_sequence_policy([], hash_move_uci=None, probs={}), [])


class TestFusionReorderCorrectness(unittest.TestCase):
    def setUp(self):
        self.moves = generate_legal_moves(Position(_MIDDLEGAME_FEN))
        self.config = FusionConfig()

    def test_output_is_exact_permutation_of_input(self):
        rng = random.Random(20260810)
        for _ in range(20):
            epoch4_probs = {lm.uci: rng.random() for lm in rng.sample(self.moves, k=rng.randint(0, len(self.moves)))}
            sequence_probs = {lm.uci: rng.random() for lm in rng.sample(self.moves, k=rng.randint(0, len(self.moves)))}
            result = reorder_with_fusion(self.moves, None, epoch4_probs, sequence_probs, self.config)
            self.assertEqual(len(result), len(self.moves))
            self.assertEqual({lm.uci for lm in result}, {lm.uci for lm in self.moves})

    def test_union_coverage_move_present_in_only_one_source_is_still_reordered(self):
        a, b, c = self.moves[0].uci, self.moves[1].uci, self.moves[2].uci
        epoch4_probs = {a: 0.9}       # solo epoch4 conoce `a`
        sequence_probs = {b: 0.9}     # solo sequence conoce `b`
        result = reorder_with_fusion(self.moves, None, epoch4_probs, sequence_probs, self.config)
        covered_front = {lm.uci for lm in result[:2]}
        self.assertEqual(covered_front, {a, b})
        self.assertNotIn(c, covered_front)  # `c` no cubierto por ninguna fuente, no debe adelantarse

    def test_move_ranked_well_by_both_sources_beats_move_ranked_well_by_only_one(self):
        moves_ucis = [lm.uci for lm in self.moves]
        best_by_both, best_by_epoch4_only = moves_ucis[0], moves_ucis[1]
        epoch4_probs = {best_by_both: 0.9, best_by_epoch4_only: 0.8}
        sequence_probs = {best_by_both: 0.9}
        result = reorder_with_fusion(self.moves, None, epoch4_probs, sequence_probs, self.config)
        self.assertLess(
            [lm.uci for lm in result].index(best_by_both),
            [lm.uci for lm in result].index(best_by_epoch4_only),
        )

    def test_never_combines_raw_indices_only_uci_space(self):
        # Prueba indirecta de la regla "combinacion siempre en espacio
        # UCI": pasar dicts con las MISMAS claves UCI pero construidos
        # desde "vocabularios" conceptualmente distintos (Epoch4 vs
        # Sequence) debe funcionar sin ningun mapeo cruzado de indices -
        # reorder_with_fusion nunca mira nada que no sea el string UCI.
        uci = self.moves[0].uci
        result = reorder_with_fusion(self.moves, None, {uci: 0.5}, {uci: 0.999999}, self.config)
        self.assertEqual(result[0].uci, uci)


# ---------------------------------------------------------------------
# Tier 1 - invariantes de sequence_policy_inference.evaluator (puras, sin
# cargar el checkpoint real)
# ---------------------------------------------------------------------
class TestHistoryValidation(unittest.TestCase):
    def test_empty_history_matches_initial_board(self):
        self.assertTrue(history_matches_board(chess.Board(), []))

    def test_history_after_e2e4_matches_board_after_e2e4(self):
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertTrue(history_matches_board(board, ["e2e4"]))

    def test_initial_board_with_e2e4_history_does_not_match(self):
        # el caso exactamente senalado en la revision: "posicion inicial +
        # ['e2e4']" es UNA POSICION DISTINTA de "board tras e2e4 +
        # ['e2e4']" - debe rechazarse, nunca aceptarse en silencio.
        self.assertFalse(history_matches_board(chess.Board(), ["e2e4"]))

    def test_illegal_move_in_history_does_not_match(self):
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertFalse(history_matches_board(board, ["e2e5"]))  # e2e5 ilegal (e2 no puede llegar a e5 de un salto)

    def test_replay_history_ucis_raises_on_illegal_move(self):
        with self.assertRaises(SequencePolicyHistoryMismatchError):
            replay_history_ucis(["e2e5"])

    def test_replay_history_ucis_raises_on_malformed_uci(self):
        with self.assertRaises(SequencePolicyHistoryMismatchError):
            replay_history_ucis(["not-a-uci"])


class TestBuildInputActionIds(unittest.TestCase):
    def test_empty_history_is_just_bos(self):
        self.assertEqual(build_input_action_ids([], context_length=256), [BOS_STD_ID])

    def test_never_appends_eos_for_predicting_a_live_move(self):
        ids = build_input_action_ids(["e2e4", "e7e5", "g1f3"], context_length=256)
        # BOS + 3 jugadas = 4 tokens, ninguno debe ser EOS (EOS es
        # exclusivo de partidas TERMINADAS, nunca al predecir la proxima
        # jugada de una posicion viva).
        self.assertEqual(len(ids), 4)
        self.assertEqual(ids[0], BOS_STD_ID)

    def test_long_history_truncated_to_exactly_context_length(self):
        long_history = ["g1f3", "g8f6"] * 200  # 400 jugadas, muy por encima de 256
        ids = build_input_action_ids(long_history, context_length=256)
        self.assertEqual(len(ids), 256)

    def test_short_history_not_truncated(self):
        ids = build_input_action_ids(["e2e4"], context_length=256)
        self.assertEqual(len(ids), 2)


# ---------------------------------------------------------------------
# Tier 1 - scope root-only REALMENTE forzado (no solo declarado)
# ---------------------------------------------------------------------
class TestScopeRootOnlyEnforced(unittest.TestCase):
    def test_sequence_never_evaluated_below_root(self):
        evaluator = _StubSequenceEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}})
        p = Position(_MIDDLEGAME_FEN)
        ctx = SearchContext(SearchLimits(max_depth=4), RepetitionTracker())
        pvs(
            p, 4, -INFINITY, INFINITY, ctx,
            sequence_policy_evaluator=evaluator,
            sequence_policy_config=SequencePolicyConfig(enabled=True),
            sequence_policy_cache=SequencePolicyCache(),
            root_move_history_ucis=[],
        )
        # a pesar de una busqueda de varios plies de profundidad, el
        # evaluador de Sequence Policy solo debe haberse llamado UNA vez
        # (en la raiz) - nunca en los nodos descendientes.
        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(ctx.sequence_positions_reordered, 1)

    def test_config_rejects_non_root_only_scope(self):
        with self.assertRaises(ValueError):
            SequencePolicyConfig(scope="all_nodes")


# ---------------------------------------------------------------------
# Tier 1 - desactivado por defecto + validaciones fail-fast
# ---------------------------------------------------------------------
class TestSequencePolicyDisabledByDefault(unittest.TestCase):
    def test_config_default_is_disabled(self):
        self.assertFalse(SequencePolicyConfig().enabled)

    def test_iterative_deepening_requires_evaluator_when_enabled(self):
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                Position(_OPENING_FEN), SearchLimits(max_depth=1), algorithm="alpha_beta", use_pvs=True,
                sequence_policy_config=SequencePolicyConfig(enabled=True),
                root_move_history_ucis=[],
            )

    def test_iterative_deepening_requires_history_when_enabled(self):
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                Position(_OPENING_FEN), SearchLimits(max_depth=1), algorithm="alpha_beta", use_pvs=True,
                sequence_policy_config=SequencePolicyConfig(enabled=True),
                sequence_policy_evaluator=_StubSequenceEvaluator(),
                root_move_history_ucis=None,
            )

    def test_iterative_deepening_rejects_sequence_config_under_negamax(self):
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                Position(_OPENING_FEN), SearchLimits(max_depth=1), algorithm="negamax",
                sequence_policy_config=SequencePolicyConfig(enabled=True),
            )

    def test_both_enabled_without_fusion_is_rejected(self):
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                Position(_OPENING_FEN), SearchLimits(max_depth=1), algorithm="alpha_beta", use_pvs=True,
                epoch4_config=Epoch4OrderingConfig(enabled=True),
                epoch4_evaluator=_StubEpoch4Evaluator(),  # duck-typed suficiente (no se llega a usar)
                sequence_policy_config=SequencePolicyConfig(enabled=True),
                sequence_policy_evaluator=_StubSequenceEvaluator(),
                root_move_history_ucis=[],
                fusion_config=None,
            )

    def test_both_enabled_with_fusion_disabled_is_also_rejected(self):
        with self.assertRaises(ValueError):
            iterative_deepening_search(
                Position(_OPENING_FEN), SearchLimits(max_depth=1), algorithm="alpha_beta", use_pvs=True,
                epoch4_config=Epoch4OrderingConfig(enabled=True),
                epoch4_evaluator=_StubEpoch4Evaluator(),
                sequence_policy_config=SequencePolicyConfig(enabled=True),
                sequence_policy_evaluator=_StubSequenceEvaluator(),
                root_move_history_ucis=[],
                fusion_config=FusionConfig(enabled=False),
            )


# ---------------------------------------------------------------------
# Tier 1 - cadena de fallback (4 escenarios)
# ---------------------------------------------------------------------
class TestFallbackChain(unittest.TestCase):
    def _fresh_context(self):
        return SearchContext(SearchLimits(max_depth=1), RepetitionTracker())

    def test_sequence_fails_falls_back_to_epoch4_only(self):
        p = Position(_MIDDLEGAME_FEN)
        ctx = self._fresh_context()
        pvs(
            p, 1, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=_StubEpoch4Evaluator(probs_by_fen={_MIDDLEGAME_FEN: {}}),
            epoch4_config=Epoch4OrderingConfig(enabled=True),
            sequence_policy_evaluator=_RaisingEvaluator(),
            sequence_policy_config=SequencePolicyConfig(enabled=True),
            sequence_policy_cache=SequencePolicyCache(),
            root_move_history_ucis=[],
        )
        self.assertEqual(ctx.sequence_fallback_triggered, 1)
        self.assertEqual(ctx.epoch4_fallback_triggered, 0)
        self.assertEqual(ctx.epoch4_positions_reordered, 1)
        self.assertEqual(ctx.fusion_positions_reordered, 0)

    def test_epoch4_fails_falls_back_to_sequence_only(self):
        p = Position(_MIDDLEGAME_FEN)
        ctx = self._fresh_context()
        pvs(
            p, 1, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=_RaisingEvaluator(),
            epoch4_config=Epoch4OrderingConfig(enabled=True),
            sequence_policy_evaluator=_StubSequenceEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}}),
            sequence_policy_config=SequencePolicyConfig(enabled=True),
            sequence_policy_cache=SequencePolicyCache(),
            root_move_history_ucis=[],
        )
        self.assertEqual(ctx.epoch4_fallback_triggered, 1)
        self.assertEqual(ctx.sequence_fallback_triggered, 0)
        self.assertEqual(ctx.sequence_positions_reordered, 1)
        self.assertEqual(ctx.fusion_positions_reordered, 0)

    def test_both_fail_falls_back_to_classical_order(self):
        p = Position(_MIDDLEGAME_FEN)
        ctx = self._fresh_context()
        pvs(
            p, 1, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=_RaisingEvaluator(),
            epoch4_config=Epoch4OrderingConfig(enabled=True),
            sequence_policy_evaluator=_RaisingEvaluator(),
            sequence_policy_config=SequencePolicyConfig(enabled=True),
            sequence_policy_cache=SequencePolicyCache(),
            root_move_history_ucis=[],
        )
        self.assertEqual(ctx.epoch4_fallback_triggered, 1)
        self.assertEqual(ctx.sequence_fallback_triggered, 1)
        self.assertEqual(ctx.epoch4_positions_reordered, 0)
        self.assertEqual(ctx.sequence_positions_reordered, 0)
        self.assertEqual(ctx.fusion_positions_reordered, 0)

    def test_both_succeed_with_fusion_enabled_uses_fusion(self):
        p = Position(_MIDDLEGAME_FEN)
        ctx = self._fresh_context()
        pvs(
            p, 1, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=_StubEpoch4Evaluator(probs_by_fen={_MIDDLEGAME_FEN: {}}),
            epoch4_config=Epoch4OrderingConfig(enabled=True),
            sequence_policy_evaluator=_StubSequenceEvaluator(probs_by_fen={_MIDDLEGAME_FEN: {}}),
            sequence_policy_config=SequencePolicyConfig(enabled=True),
            sequence_policy_cache=SequencePolicyCache(),
            root_move_history_ucis=[],
            fusion_config=FusionConfig(enabled=True),
        )
        self.assertEqual(ctx.fusion_positions_reordered, 1)
        self.assertEqual(ctx.epoch4_positions_reordered, 0)
        self.assertEqual(ctx.sequence_positions_reordered, 0)
        self.assertEqual(ctx.epoch4_fallback_triggered, 0)
        self.assertEqual(ctx.sequence_fallback_triggered, 0)

    def test_epoch4_failure_alone_does_not_crash_search_regression(self):
        # Retrofit de Fase VIII sobre el bloque de Epoch4 construido en
        # Fase VII (antes no estaba protegido con try/except) - una falla
        # de Epoch4 SOLO (sin Sequence Policy involucrado) no debe tumbar
        # la busqueda, debe degradar a orden clasico.
        p = Position(_MIDDLEGAME_FEN)
        ctx = self._fresh_context()
        score, pv = pvs(
            p, 1, -INFINITY, INFINITY, ctx,
            epoch4_evaluator=_RaisingEvaluator(),
            epoch4_config=Epoch4OrderingConfig(enabled=True),
        )
        self.assertEqual(ctx.epoch4_fallback_triggered, 1)
        self.assertEqual(ctx.epoch4_positions_reordered, 0)
        self.assertIsNotNone(pv)  # la busqueda sigue produciendo una PV valida


# ---------------------------------------------------------------------
# Tier 1 - aislamiento de latencia
# ---------------------------------------------------------------------
class TestSequenceLatencyIsolation(unittest.TestCase):
    def test_inference_time_scales_with_predict_calls_not_cache_hits(self):
        sleep_s = 0.02
        evaluator = _StubSequenceEvaluator(probs_by_fen={_OPENING_FEN: {}}, sleep_seconds=sleep_s)
        cache = SequencePolicyCache()
        ctx = SearchContext(SearchLimits(max_depth=1), RepetitionTracker())
        p = Position(_OPENING_FEN)

        evaluate_sequence_policy(p, evaluator, cache, [], ctx)
        for _ in range(5):
            evaluate_sequence_policy(p, evaluator, cache, [], ctx)

        self.assertEqual(evaluator.predict_calls, 1)
        self.assertEqual(ctx.sequence_cache_hits, 5)
        self.assertLess(ctx.sequence_inference_time_ms, sleep_s * 1000 * 3)


# ---------------------------------------------------------------------
# Tier 1.5 - modelo real (skip-si-no-disponible)
# ---------------------------------------------------------------------
class TestSequencePolicyWithRealEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CHECKPOINT_PATH.exists():
            raise unittest.SkipTest(f"checkpoint Sequence Policy no encontrado en {_CHECKPOINT_PATH}")
        from sequence_policy_inference.evaluator import SequencePolicyEvaluator
        cls.evaluator = SequencePolicyEvaluator(_CHECKPOINT_PATH)

    def test_checkpoint_sha256_matches_documented_value(self):
        # fallo DURO (no skip) - trip-wire permanente contra corrupcion o
        # modificacion accidental del checkpoint congelado.
        import hashlib
        h = hashlib.sha256()
        with open(_CHECKPOINT_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), _EXPECTED_CHECKPOINT_SHA256)

    def test_legal_mask_correctness_on_initial_position(self):
        board = chess.Board()
        probs = self.evaluator.predict(board, [])
        legal_ucis = {m.uci() for m in board.legal_moves}
        self.assertTrue(set(probs.keys()).issubset(legal_ucis))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_in_distribution_case_board_after_e2e4(self):
        # caso EN DISTRIBUCION real: board DESPUES de e2e4 + history=['e2e4']
        # (no "posicion inicial + ['e2e4']", que es una posicion distinta -
        # ver TestHistoryValidation.test_initial_board_with_e2e4_history_does_not_match)
        board = chess.Board()
        board.push_uci("e2e4")
        probs = self.evaluator.predict(board, ["e2e4"])
        legal_ucis = {m.uci() for m in board.legal_moves}
        self.assertTrue(set(probs.keys()).issubset(legal_ucis))
        self.assertGreater(len(probs), 0)

    def test_long_real_history_does_not_crash(self):
        random.seed(0)
        board = chess.Board()
        history = []
        while len(history) < 260 and not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            history.append(move.uci())
            board.push(move)
        self.assertGreater(len(history), 255, "el juego aleatorio termino antes de superar 255 jugadas")
        probs = self.evaluator.predict(board, history)
        legal_ucis = {m.uci() for m in board.legal_moves}
        self.assertTrue(set(probs.keys()).issubset(legal_ucis))

    def test_incompatible_history_raises_controlled_error(self):
        board = chess.Board()
        with self.assertRaises(SequencePolicyHistoryMismatchError):
            self.evaluator.predict(board, ["e2e4"])  # board inicial, historia dice que ya se jugo e2e4

    def test_reordering_is_genuinely_different_from_classical_for_at_least_one_fen(self):
        cache = SequencePolicyCache()
        any_reordered = False
        for fen in (_OPENING_FEN,):
            # unico FEN de _TIER2_SMOKE_FENS con historial real conocido
            # desde el inicio estandar (ver limitacion documentada en
            # TestSequencePolicyWithRealEvaluator - el resto de
            # _TIER2_SMOKE_FENS no tiene trayectoria reconstruible).
            board = chess.Board()
            board.push_uci("e2e4")
            p = Position(board.fen())
            legal_moves = generate_legal_moves(p)
            probs = evaluate_sequence_policy(p, self.evaluator, cache, ["e2e4"])
            reordered = reorder_with_sequence_policy(legal_moves, hash_move_uci=None, probs=probs)
            if [lm.uci for lm in reordered] != [lm.uci for lm in legal_moves]:
                any_reordered = True
        self.assertTrue(any_reordered, "Sequence Policy nunca reordeno nada - wiring probablemente no-op")


# ---------------------------------------------------------------------
# Tier 2 - regresion de score, modos A/B/C/D (stub, sin depender de
# historia real - los stubs ignoran root_move_history_ucis)
# ---------------------------------------------------------------------
class TestModesNeverChangeTacticalTruth(unittest.TestCase):
    """Mismo criterio ya establecido (ver TestEpoch4NeverChangesTacticalTruth
    en test_epoch4_integration.py): SCORE es el invariante real y
    garantizado por construccion (PVS de ancho completo es order-invariant
    en el valor minimax, con o sin empates) - best_move literal NO es el
    criterio bajo empates de score."""

    def test_score_identical_across_modes_a_b_c_d(self):
        rng = random.Random(20260810)
        for fen in _TIER2_SMOKE_FENS:
            with self.subTest(fen=fen):
                baseline = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                )

                legal = generate_legal_moves(Position(fen))
                epoch4_probs = {lm.uci: rng.random() for lm in legal}
                sequence_probs = {lm.uci: rng.random() for lm in legal}
                epoch4_ev = _StubEpoch4Evaluator(probs_by_fen={fen: epoch4_probs})
                sequence_ev = _StubSequenceEvaluator(probs_by_fen={fen: sequence_probs})

                # Modo B - solo Epoch4
                mode_b = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=epoch4_ev, epoch4_config=Epoch4OrderingConfig(enabled=True),
                )
                self.assertEqual(mode_b.score, baseline.score)

                # Modo C - solo Sequence Policy
                mode_c = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    sequence_policy_evaluator=sequence_ev, sequence_policy_config=SequencePolicyConfig(enabled=True),
                    root_move_history_ucis=[],
                )
                self.assertEqual(mode_c.score, baseline.score)

                # Modo D - fusion Epoch4 + Sequence Policy
                mode_d = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=epoch4_ev, epoch4_config=Epoch4OrderingConfig(enabled=True),
                    sequence_policy_evaluator=sequence_ev, sequence_policy_config=SequencePolicyConfig(enabled=True),
                    root_move_history_ucis=[], fusion_config=FusionConfig(enabled=True),
                )
                self.assertEqual(mode_d.score, baseline.score)

    def test_historical_forced_defense_score_unaffected_across_modes(self):
        cases = [
            ("1kr5/1p3p1p/8/4Q3/1P1RPP2/8/PKP1q3/8 b - - 8 35", "b8a7"),
            ("5R2/4N1pk/1p3n1p/1P5P/8/r6K/5PP1/8 w - - 13 44", "f2f3"),
        ]
        for fen, solution in cases:
            with self.subTest(fen=fen):
                baseline = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                )
                self.assertEqual(baseline.best_move, solution)  # control tie-free

                legal = generate_legal_moves(Position(fen))
                adversarial_epoch4 = {lm.uci: (0.99 if lm.uci != solution else 0.01) for lm in legal}
                adversarial_sequence = {lm.uci: (0.99 if lm.uci != solution else 0.01) for lm in legal}

                mode_d = iterative_deepening_search(
                    Position(fen), SearchLimits(max_depth=2), algorithm="alpha_beta", use_pvs=True,
                    epoch4_evaluator=_StubEpoch4Evaluator(probs_by_fen={fen: adversarial_epoch4}),
                    epoch4_config=Epoch4OrderingConfig(enabled=True),
                    sequence_policy_evaluator=_StubSequenceEvaluator(probs_by_fen={fen: adversarial_sequence}),
                    sequence_policy_config=SequencePolicyConfig(enabled=True),
                    root_move_history_ucis=[], fusion_config=FusionConfig(enabled=True),
                )
                self.assertEqual(mode_d.score, baseline.score)


if __name__ == "__main__":
    unittest.main()
