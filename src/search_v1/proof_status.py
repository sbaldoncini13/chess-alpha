"""Semantica de resultados de prueba de search_v1 (Fase VI).

NO se usa una clasificacion binaria mate/no-mate: cuando una busqueda de
prueba (DFPN/mate_search/forced_defense) se agota por presupuesto (nodos/
tiempo/profundidad), el resultado es UNKNOWN con el motivo EXACTO - nunca
se afirma "no existe mate" cuando solo se agoto el presupuesto (item
explicito del pedido de Fase VI, y la razon de ser de esta enumeracion
separada de `search_v1.terminal.TerminalStatus`, que describe posiciones,
no resultados de PRUEBA)."""

from __future__ import annotations

from enum import Enum, auto


class ProofStatus(Enum):
    # --- Pruebas EXACTAS, completas ---
    PROVEN_MATE = auto()                       # existe una linea forzada que termina en jaque mate
    PROVEN_NO_MATE = auto()                     # se demostro EXHAUSTIVAMENTE que no existe mate forzado
    PROVEN_NO_MATE_WITHIN_BOUND = auto()        # no existe mate forzado DENTRO de un limite de plies explicito (no es una prueba de "nunca", solo "no en este horizonte")

    # --- Resultados de posiciones terminales encontradas DURANTE la prueba ---
    TERMINAL_WIN = auto()          # jaque mate a favor del atacante (equivalente a PROVEN_MATE en un nodo hoja)
    TERMINAL_LOSS = auto()         # jaque mate en contra del atacante
    TERMINAL_DRAW = auto()         # ahogado / material insuficiente en un nodo hoja
    DRAW_BY_REPETITION = auto()    # repeticion (triple contra el historial real, o ciclo dentro del propio camino de busqueda - ver dfpn.py)
    DRAW_BY_50_MOVE = auto()
    DRAW_BY_INSUFFICIENT_MATERIAL = auto()

    # --- Presupuesto agotado ANTES de completar la prueba - NUNCA se
    # afirma "no hay mate" en estos casos, el motivo queda explicito ---
    UNKNOWN_NODE_LIMIT = auto()
    UNKNOWN_TIME_LIMIT = auto()
    UNKNOWN_DEPTH_LIMIT = auto()


PROVEN_STATUSES = frozenset({
    ProofStatus.PROVEN_MATE, ProofStatus.PROVEN_NO_MATE, ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND,
})
DRAW_STATUSES = frozenset({
    ProofStatus.TERMINAL_DRAW, ProofStatus.DRAW_BY_REPETITION,
    ProofStatus.DRAW_BY_50_MOVE, ProofStatus.DRAW_BY_INSUFFICIENT_MATERIAL,
})
UNKNOWN_STATUSES = frozenset({
    ProofStatus.UNKNOWN_NODE_LIMIT, ProofStatus.UNKNOWN_TIME_LIMIT, ProofStatus.UNKNOWN_DEPTH_LIMIT,
})


def is_proven(status: ProofStatus) -> bool:
    return status in PROVEN_STATUSES


def is_unknown(status: ProofStatus) -> bool:
    return status in UNKNOWN_STATUSES
