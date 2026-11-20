"""Proveedor de ordenamiento Epoch4 de search_v1 (Fase VII) - consumidor
READ-ONLY del checkpoint congelado `models/selected/
policy_value_hybrid_selected_epoch4.pt` (nunca modificado desde aca).

REGLA CENTRAL (item explicito del pedido, ver SEARCH_TREE_V1_TECHNICAL_
DESIGN.md seccion 3 y SEARCH_TREE_V1_RISK_REGISTER.md R4): Epoch4 SOLO
reordena la lista de jugadas que PVS ya genero via move_ordering.py -
jamas agrega, quita o cambia el conjunto de jugadas exploradas. Esto es
deliberadamente MAS ESTRICTO que el "Policy Prior" del motor viejo
(Fase 6/7 de src/search/negamax.py, RECHAZADO Y CERRADO tras encontrar 2
casos donde ignoraba la unica defensa correcta ante mate) - ese diseno
dejaba que la policy alterara resultados; este NO puede alterarlos por
construccion (ver reorder_with_epoch4: siempre devuelve una permutacion
exacta de la entrada).

NO reimplementa carga de modelo ni mapeo indice<->UCI - reutiliza
`HybridNeuralEvaluator` (src/chess_alpha/inference.py) via INYECCION DE
DEPENDENCIA: este modulo nunca importa `chess_alpha` ni construye el
evaluador, solo recibe un objeto ya construido que expone
`.predict(board) -> (probs: dict[uci, float], value_white: float)`
(duck-typed, no se exige la clase concreta).

TRAMPA DE LATENCIA (verificada leyendo inference.py antes de escribir este
modulo): `HybridNeuralEvaluator.top_k_moves()` llama internamente a
`.predict()` una SEGUNDA vez - es la causa concreta, ya vista en
src/search/negamax.py, de una regresion de latencia real ~2.2x por doble
inferencia. `evaluate_epoch4()` de aca NUNCA llama a `.top_k_moves()` -
llama `.predict()` exactamente una vez por posicion (o cero veces si hay
cache hit) y deriva cualquier ranking del dict `probs` ya devuelto.

Cache namespace COMPLETAMENTE SEPARADO de `TranspositionTable` (Fase III)
y `DFPNTable` (Fase VI) - misma regla de aislamiento ya aplicada entre
esas dos (nunca comparten entradas), extendida aca a una tercera tabla.

SCOPE (Fase VIII, agregado ADITIVO tras medir el costo real de Epoch4
"all_nodes" en el micro-benchmark A/B/C/D - ver
reports/search_tree_v1/PHASE_VIII_MICRO_BENCHMARK_SCOPE.json): por
defecto Epoch4 se evalua en CADA nodo del arbol (`scope="all_nodes"`,
identico al comportamiento original de Fase VII, sin cambios de
comportamiento para nada que no pase `scope` explicitamente). `scope=
"root_only"` es un modo NUEVO, opt-in, que restringe la evaluacion a
`context.ply_from_root == 0` - mismo patron ya usado por Sequence Policy
(sequence_policy_provider.py) por la misma razon (costo de inferencia por
nodo). `"all_nodes"` NUNCA se elimina ni se reemplaza por default nuevo -
el pedido explicito fue "no eliminar all_nodes"."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Epoch4OrderingConfig:
    enabled: bool = False
    cache_capacity_entries: int = 65536
    coverage_top_k: int = 8  # solo telemetria de coverage@K - NUNCA trunca el reordenamiento real
    scope: str = "all_nodes"  # "all_nodes" (default, comportamiento original de Fase VII) | "root_only"

    def __post_init__(self):
        if self.cache_capacity_entries < 1:
            raise ValueError("cache_capacity_entries debe ser >= 1")
        if self.coverage_top_k < 1:
            raise ValueError("coverage_top_k debe ser >= 1")
        if self.scope not in ("all_nodes", "root_only"):
            raise ValueError(f"scope invalido: {self.scope!r} - esperado 'all_nodes' o 'root_only'")


@dataclass
class Epoch4CacheEntry:
    key: int
    probs: dict


class Epoch4OrderingCache:
    """Direct-mapped (mismo diseno auditable que TranspositionTable/
    DFPNTable: `key % capacity`, clave completa SIEMPRE comparada antes de
    aceptar un hit). Guarda UNICAMENTE `probs` (la distribucion de la
    Policy ya legal-masked/renormalizada por HybridNeuralEvaluator) - el
    value head de Epoch4 no se usa ni se cachea aca, Fase VII es
    exclusivamente ordenamiento."""

    def __init__(self, capacity_entries: int = 65536):
        if capacity_entries < 1:
            raise ValueError("capacity_entries debe ser >= 1")
        self.capacity = capacity_entries
        self._table: list[Epoch4CacheEntry | None] = [None] * capacity_entries
        self.probes = 0
        self.hits = 0
        self.collisions = 0
        self.stores = 0

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
        self._table[idx] = Epoch4CacheEntry(key=key, probs=probs)
        self.stores += 1


def evaluate_epoch4(position, evaluator, cache: Epoch4OrderingCache | None, context=None) -> dict:
    """Unico punto del modulo donde se llama `evaluator.predict()`. Prueba
    el cache primero (transposiciones dentro de la misma busqueda); en
    miss, llama `evaluator.predict(position.board)` EXACTAMENTE una vez y
    guarda solo `probs` (nunca `value_white` - no se usa en Fase VII).

    Si se pasa `context` (un SearchContext, opt-in), incrementa la
    telemetria dedicada (`epoch4_cache_probes/hits/inferences`) y acumula
    `epoch4_inference_time_ms` medido con `time.perf_counter()` aislado
    ESTRICTAMENTE alrededor de la llamada a `.predict()` - nunca incluye
    el costo de cache lookup ni de reordenar (medicion limpia, misma
    leccion ya documentada en el motor viejo tras la regresion ~2.2x)."""
    key = position.zobrist_key

    if cache is not None:
        if context is not None:
            context.epoch4_cache_probes += 1
        cached = cache.probe(key)
        if cached is not None:
            if context is not None:
                context.epoch4_cache_hits += 1
            return cached

    t0 = time.perf_counter()
    probs, _value_white = evaluator.predict(position.board)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if context is not None:
        context.epoch4_inferences += 1
        context.epoch4_inference_time_ms += elapsed_ms

    if cache is not None:
        cache.store(key, probs)

    return probs


def reorder_with_epoch4(moves: list, hash_move_uci: str | None, probs: dict) -> list:
    """Funcion PURA. Devuelve SIEMPRE una permutacion exacta de `moves`
    (mismo largo, mismo set de UCIs) - jamas agrega, quita o duplica una
    jugada, sin importar que contenga `probs` (garantiza por construccion
    la regla central del modulo, no solo por convencion).

    - El hash move, si esta en indice 0 de `moves` (ya puesto ahi por
      move_ordering.py), queda FIJO ahi - Epoch4 nunca lo desplaza (el
      hash move ya es la senal de ordenamiento mas barata y confiable
      disponible, no hay razon para cederle el lugar).
    - El resto se parte en `covered` (uci presente en `probs`, ordenado
      por probabilidad DESCENDENTE) y `uncovered` (uci ausente de
      `probs` - vocabulario de Epoch4 no lo cubre) - `uncovered` conserva
      el orden CLASICO original sin tocar, este ES el fallback
      obligatorio de vocabulario incompleto (riesgo R6): un movimiento
      legal fuera del vocabulario de Epoch4 nunca se pierde ni se
      reordena a ciegas, simplemente sigue exactamente donde el
      ordenamiento clasico (MVV-LVA/killers/history) ya lo puso."""
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

    assert len(result) == len(moves), "reorder_with_epoch4 no debe cambiar la cantidad de jugadas"
    assert {lm.uci for lm in result} == {lm.uci for lm in moves}, "reorder_with_epoch4 no debe cambiar el conjunto de jugadas"

    return result
