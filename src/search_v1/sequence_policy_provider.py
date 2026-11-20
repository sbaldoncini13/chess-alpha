"""Proveedor de ordenamiento Sequence Policy V1 FULL de search_v1 (Fase
VIII) - consumidor READ-ONLY del checkpoint congelado
`Chess_Alpha_Sequence_Policy_FULL_2286_PC\\last.ckpt` (nunca modificado
desde aca) y, indirectamente via `sequence_policy_inference.evaluator`,
del paquete `sequence_policy_v1/src/sequence_policy/` (tambien solo
lectura).

REGLA CENTRAL (identica a Epoch4, ver epoch4_provider.py): Sequence Policy
SOLO reordena la lista de jugadas que PVS ya genero - jamas agrega, quita
o cambia el conjunto de jugadas exploradas.

SCOPE = root_only (decision de diseno explicita de Fase VIII, no un
placeholder temporal): `RepetitionTracker` (Fase I/II, frozen) solo
trackea zobrist keys, no historial de jugadas UCI - extender a nodos no-
raiz requeriria hilar una pila de historial mutable por las 5 llamadas
recursivas de pvs.py (complejidad real, evitable). Ademas, correr un
Transformer de 8 capas en cada nodo del arbol seria prohibitivamente caro
comparado con la CNN de Epoch4. `SequencePolicyConfig.scope` existe como
campo explicito para una extension futura, pero SOLO valida
`"root_only"` - cualquier otro valor es un error, no un modo soportado.

`search_v1` NUNCA importa `sequence_policy_inference` ni
`sequence_policy_v1` directamente - solo recibe, por inyeccion de
dependencia, un objeto ya construido que expone
`.predict(board, history_ucis) -> dict[uci, float]` (duck-typed, no se
exige la clase concreta)."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SequencePolicyConfig:
    enabled: bool = False
    cache_capacity_entries: int = 65536
    scope: str = "root_only"

    def __post_init__(self):
        if self.cache_capacity_entries < 1:
            raise ValueError("cache_capacity_entries debe ser >= 1")
        if self.scope != "root_only":
            raise ValueError(
                f"scope invalido: {self.scope!r} - solo 'root_only' esta soportado en Fase VIII "
                "(scope no-raiz requeriria hilar historial de jugadas por toda la recursion de pvs.py, "
                "no implementado deliberadamente)"
            )


@dataclass
class SequencePolicyCacheEntry:
    key: int
    probs: dict


class SequencePolicyCache:
    """Direct-mapped (mismo diseno que Epoch4OrderingCache/DFPNTable/
    TranspositionTable), namespace COMPLETAMENTE SEPARADO de las otras
    tres tablas - una CUARTA tabla, nunca comparte entradas.

    CLAVE DELIBERADAMENTE DISTINTA de Epoch4OrderingCache: acá se combina
    `position.zobrist_key` CON el historial de jugadas de la raiz
    (`root_move_history_ucis`), no solo la posicion. Razon: la premisa
    completa de Sequence Policy (ver docs/PROJECT_STATE.md seccion B) es
    que dos posiciones IDENTICAS alcanzadas por historias DISTINTAS son
    consultas distintas para este modelo (transposicion/plan/repeticion
    cuentan como contexto). Con scope=root_only y una cache nueva por
    llamada a iterative_deepening_search esto seria "seguro por
    coincidencia" incluso con clave zobrist-only (la raiz no cambia
    dentro de una misma llamada) - la clave compuesta lo hace correcto
    POR CONSTRUCCION, no por la casualidad del uso actual."""

    def __init__(self, capacity_entries: int = 65536):
        if capacity_entries < 1:
            raise ValueError("capacity_entries debe ser >= 1")
        self.capacity = capacity_entries
        self._table: list[SequencePolicyCacheEntry | None] = [None] * capacity_entries
        self.probes = 0
        self.hits = 0
        self.collisions = 0
        self.stores = 0

    @staticmethod
    def compute_key(zobrist_key: int, root_move_history_ucis) -> int:
        return hash((zobrist_key, tuple(root_move_history_ucis)))

    def _index(self, key: int) -> int:
        return key % self.capacity

    def probe(self, key: int) -> dict | None:
        self.probes += 1
        idx = self._index(key)
        entry = self._table[idx]
        if entry is None:
            return None
        if entry.key != key:
            self.collisions += 1
            return None
        self.hits += 1
        return entry.probs

    def store(self, key: int, probs: dict) -> None:
        idx = self._index(key)
        self._table[idx] = SequencePolicyCacheEntry(key=key, probs=probs)
        self.stores += 1


def evaluate_sequence_policy(position, evaluator, cache: SequencePolicyCache | None,
                              root_move_history_ucis, context=None) -> dict:
    """Unico punto del modulo donde se llama `evaluator.predict()`. Prueba
    el cache primero (clave compuesta zobrist+historia); en miss, llama
    `evaluator.predict(position.board, root_move_history_ucis)` EXACTAMENTE
    una vez. Latencia medida con `time.perf_counter()` aislada
    estrictamente alrededor de esa llamada (misma disciplina que
    evaluate_epoch4 - nunca incluye cache lookup ni reordenar)."""
    key = SequencePolicyCache.compute_key(position.zobrist_key, root_move_history_ucis)

    if cache is not None:
        if context is not None:
            context.sequence_cache_probes += 1
        cached = cache.probe(key)
        if cached is not None:
            if context is not None:
                context.sequence_cache_hits += 1
            return cached

    t0 = time.perf_counter()
    probs = evaluator.predict(position.board, root_move_history_ucis)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if context is not None:
        context.sequence_inferences += 1
        context.sequence_inference_time_ms += elapsed_ms

    if cache is not None:
        cache.store(key, probs)

    return probs


def reorder_with_sequence_policy(moves: list, hash_move_uci: str | None, probs: dict) -> list:
    """Funcion PURA - mismo contrato exacto que
    epoch4_provider.reorder_with_epoch4 (permutacion exacta garantizada,
    hash move fijo en indice 0, cubiertas ordenadas por probabilidad
    descendente, no-cubiertas mantienen el orden clasico original sin
    tocar). Implementada de forma independiente (sin importar de
    epoch4_provider.py) para que cada proveedor sea desactivable/testeable
    sin acoplamiento cruzado entre los dos modulos.

    A diferencia de Epoch4 (vocabulario fijo de 1.868 con huecos de
    cobertura reales), el codec de Sequence Policy es cerrado y universal
    (accion = from/to/promo) - en la practica `probs` deberia cubrir el
    100% de las jugadas legales, así que la rama "no-cubiertas" debería
    quedar vacía casi siempre; si no lo está, es señal de alerta (ver
    telemetria `sequence_moves_uncovered_total`), no un fallback rutinario."""
    if not moves:
        return []

    pinned = None
    rest = moves
    if moves[0].uci == hash_move_uci:
        pinned = moves[0]
        rest = moves[1:]

    covered = [lm for lm in rest if lm.uci in probs]
    uncovered = [lm for lm in rest if lm.uci not in probs]
    covered.sort(key=lambda lm: probs[lm.uci], reverse=True)

    result = ([pinned] if pinned is not None else []) + covered + uncovered

    assert len(result) == len(moves), "reorder_with_sequence_policy no debe cambiar la cantidad de jugadas"
    assert {lm.uci for lm in result} == {lm.uci for lm in moves}, "reorder_with_sequence_policy no debe cambiar el conjunto de jugadas"

    return result
