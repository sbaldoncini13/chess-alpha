"""Resultado estructurado de busqueda de search_v1 (Fase II).

`elapsed_ms`/`nodes_per_second` se calculan con la precision REAL medida
(time.perf_counter, reloj monotonico) - no se inventa una precision mayor
a la que el reloj realmente ofrece; nodes_per_second es 0.0 (no
division por cero) cuando elapsed_ms es 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ScoreType(Enum):
    CENTIPAWNS = auto()
    MATE = auto()


class StopReason(Enum):
    DEPTH_COMPLETE = auto()   # iterative deepening llego a requested_depth normalmente
    MAX_NODES = auto()
    MAX_TIME = auto()
    TERMINAL_ROOT = auto()    # la posicion raiz ya era terminal, no hubo busqueda
    MAX_QNODES = auto()       # Fase IV: techo global de nodos de quiescence alcanzado
    MAX_CHECK_EXTENSIONS = auto()  # Fase IV: presupuesto de extensiones de jaque agotado en quiescence


@dataclass
class SearchResult:
    best_move: str | None
    score: int
    score_type: ScoreType
    mate_distance_plies: int | None
    principal_variation: list[str]
    completed_depth: int
    requested_depth: int
    nodes: int
    leaf_nodes: int
    terminal_nodes: int
    alpha_beta_cutoffs: int
    elapsed_ms: float
    nodes_per_second: float
    stop_reason: StopReason
    completed: bool
    legal: bool
    metadata: dict = field(default_factory=dict)
