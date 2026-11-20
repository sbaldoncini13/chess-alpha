"""Search Tree Observatory - instrumentacion OPT-IN, ADITIVA, por-nodo de
`pvs()`/`quiescence()`/`iterative_deepening_search()` (PASO 0 de la
iniciativa de observabilidad, ver reports/search_tree_v1/
OBSERVATORY_AUDIT_MATRIX.md).

REGLA DE ORO (verificada con tests, ver test_observer.py): con
`context.observer=None` (default, SIEMPRE el valor salvo que quien llama
pida tracing explicitamente), el unico costo agregado en cada punto de
instrumentacion es un chequeo `if context.observer is not None:` - CERO
cambio de comportamiento, CERO cambio de score/best_move/nodes. Ningun
componente FROZEN (PVS, quiescence, TT, ordering, DFPN, Epoch4, Sequence)
fue modificado en su LOGICA - solo se agregaron llamadas de observacion
de solo-lectura en puntos ya existentes.

Este modulo NO inventa "pesos": el mecanismo real (hash move=prioridad,
killer=flag, history=score numerico, Epoch4/Sequence=probabilidad,
Fusion=rank RRF) se registra tal cual sale de move_ordering.py/
epoch4_provider.py/sequence_policy_provider.py/candidate_fusion.py, nunca
se reinterpreta como un peso lineal ficticio."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class NodeEvent:
    """Un nodo observado. Los campos de "exit" quedan en None hasta que
    `SearchObserver.exit()` los completa - un evento nunca deberia
    exportarse con exit_recorded=False (indicaria una excepcion no
    contemplada, no un dato faltante intencional)."""

    node_id: int
    parent_id: int | None
    node_type: str  # "search" (pvs) o "quiescence"
    ply: int
    depth_remaining: int
    side_to_move: str
    fen: str
    move_from_parent: str | None
    alpha_in: int
    beta_in: int
    window: str  # "full" o "null", derivado de (beta_in - alpha_in) > 1

    # completados por record_move_list() - None si el nodo termino antes
    # de llegar a generar/ordenar jugadas (terminal, TT-cutoff, etc.)
    legal_move_count: int | None = None
    search_order: list[str] | None = None
    hash_move: str | None = None

    # completados por record_tt_status()
    tt_status: str = "not_queried"  # not_queried|miss|hit_exact|hit_lower_bound|hit_upper_bound

    # completados por record_cutoff() - None si el nodo no genero cutoff
    cutoff_move_uci: str | None = None
    cutoff_move_index: int | None = None

    # completados por exit() - alpha_out/beta_out son DERIVADOS del
    # invariante fail-soft real del codigo (verificado leyendo pvs.py/
    # quiescence.py: beta nunca se reasigna dentro del loop de un nodo,
    # alpha solo aumenta via `if score > alpha: alpha = score`) - no son
    # un dato capturado directamente del interior de la funcion, pero
    # tampoco un valor inventado: se derivan de una propiedad demostrable
    # del algoritmo, documentada en _classify().
    exit_recorded: bool = False
    alpha_out: int | None = None
    beta_out: int | None = None
    returned_score: int | None = None
    principal_variation: list[str] = field(default_factory=list)
    elapsed_ms: float | None = None
    node_class: str | None = None  # PV|CUT|ALL, derivado post-hoc (ver _classify)

    _t0: float = field(default=0.0, repr=False)


def _classify(alpha_in: int, beta_in: int, returned_score: int) -> str:
    """Clasificacion PV/CUT/ALL ESTANDAR de alpha-beta, derivada de datos
    reales (alpha_in/beta_in/returned_score) - no es un campo que el motor
    produzca nativamente (confirmado en la auditoria, ver
    OBSERVATORY_AUDIT_MATRIX.md fila 6), pero tampoco es un valor
    inventado: es la definicion de texto de busqueda estandar (Knuth/Moore
    1975) aplicada a datos reales de este nodo.
        CUT node: returned_score >= beta_in (fail-high, produjo corte)
        ALL node: returned_score <= alpha_in (fail-low, se busco todo)
        PV node:  alpha_in < returned_score < beta_in (score exacto)"""
    if returned_score >= beta_in:
        return "CUT"
    if returned_score <= alpha_in:
        return "ALL"
    return "PV"


class SearchObserver:
    """Se asigna a `context.observer` (ver search_context.py). Un observer
    nuevo por busqueda (no se reutiliza entre llamadas a
    iterative_deepening_search, igual que `tt`/`ordering_state` NO se
    reutilizan con ese proposito salvo que quien llama lo pida
    explicitamente)."""

    def __init__(self, max_events: int | None = None):
        """`max_events`: tope de nodos a registrar en detalle (control de
        tamano del arbol, item 7 del pedido) - por encima del tope se
        siguen contando pero NO se guarda el detalle completo (evita
        traces de cientos de miles de nodos en memoria). None = sin tope
        (razonable a depth<=4/5, ver auditoria: ~1600-3700 nodos medidos)."""
        self.events: list[NodeEvent] = []
        self.iterations: list[dict] = []
        self.max_events = max_events
        self._next_id = 0
        self._stack: list[int] = []
        self._truncated_count = 0

    # --- ciclo de vida de nodo ---

    def enter(self, *, node_type, position, depth_remaining, alpha, beta, ply, move_from_parent) -> int:
        node_id = self._next_id
        self._next_id += 1
        parent_id = self._stack[-1] if self._stack else None
        self._stack.append(node_id)

        if self.max_events is not None and node_id >= self.max_events:
            self._truncated_count += 1
            return node_id  # id valido para poder hacer pop() en exit(), pero sin NodeEvent guardado

        event = NodeEvent(
            node_id=node_id, parent_id=parent_id, node_type=node_type,
            ply=ply, depth_remaining=depth_remaining, side_to_move=position.side_to_move,
            fen=position.fen, move_from_parent=move_from_parent,
            alpha_in=alpha, beta_in=beta, window="full" if (beta - alpha) > 1 else "null",
        )
        event._t0 = time.perf_counter()
        self.events.append(event)
        return node_id

    def exit(self, node_id: int, score: int, pv: list[str]) -> None:
        self._stack.pop()
        if self.max_events is not None and node_id >= self.max_events:
            return  # truncado en enter(), nada que completar
        event = self.events[node_id]
        event.exit_recorded = True
        event.returned_score = score
        event.principal_variation = list(pv)
        event.elapsed_ms = (time.perf_counter() - event._t0) * 1000.0
        event.node_class = _classify(event.alpha_in, event.beta_in, score)
        # derivado del invariante fail-soft (ver comentario en NodeEvent) -
        # beta constante dentro del nodo; alpha == alpha_in si ALL (nunca
        # mejoro), si no alpha llego a valer exactamente `score` (ultima
        # mejora antes de retornar/cortar).
        event.beta_out = event.beta_in
        event.alpha_out = event.alpha_in if event.node_class == "ALL" else score

    def _current_event(self) -> NodeEvent | None:
        if not self._stack:
            return None
        node_id = self._stack[-1]
        if self.max_events is not None and node_id >= self.max_events:
            return None
        return self.events[node_id]

    # --- detalle intra-nodo, atado al nodo actualmente en el tope de la pila ---

    def record_move_list(self, moves_uci: list[str], hash_move_uci: str | None) -> None:
        ev = self._current_event()
        if ev is not None:
            ev.legal_move_count = len(moves_uci)
            ev.search_order = list(moves_uci)
            ev.hash_move = hash_move_uci

    def record_tt_status(self, status: str) -> None:
        ev = self._current_event()
        if ev is not None:
            ev.tt_status = status

    def record_cutoff(self, move_uci: str, move_index: int) -> None:
        ev = self._current_event()
        if ev is not None:
            ev.cutoff_move_uci = move_uci
            ev.cutoff_move_index = move_index

    # --- iterative deepening (no depende de la pila de nodos) ---

    def record_iteration(self, *, depth: int, score: int, pv: list[str], nodes_cumulative: int,
                          q_nodes_cumulative: int, elapsed_ms_cumulative: float, tt_hits_cumulative: int | None,
                          alpha_beta_cutoffs_cumulative: int) -> None:
        self.iterations.append({
            "depth": depth, "score": score, "pv": list(pv),
            "nodes_cumulative": nodes_cumulative, "q_nodes_cumulative": q_nodes_cumulative,
            "elapsed_ms_cumulative": elapsed_ms_cumulative, "tt_hits_cumulative": tt_hits_cumulative,
            "alpha_beta_cutoffs_cumulative": alpha_beta_cutoffs_cumulative,
        })

    # --- export ---

    def to_dict(self) -> dict:
        return {
            "events": [
                {k: v for k, v in vars(e).items() if not k.startswith("_")}
                for e in self.events
            ],
            "iterations": self.iterations,
            "n_events_recorded": len(self.events),
            "n_events_truncated": self._truncated_count,
            "max_events": self.max_events,
        }
