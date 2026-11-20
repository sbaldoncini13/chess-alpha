"""Deteccion de posiciones terminales de search_v1.

Deliberadamente separado de static_evaluation.py (item explicito del
alcance de Fase I: "valores terminales separados de la evaluacion
estatica"). Convencion adoptada: `evaluate_static()` SIEMPRE calcula
balance material, sin mirar si la posicion es terminal - quien llama
(por ahora `result.analyze_position()`, en el futuro el buscador) es
responsable de consultar `detect_terminal()` primero y, si `is_terminal`
es True, usar `TerminalResult` en vez del score material.

La regla de 50 movimientos se incluye aca (no en repetition.py) porque es
enteramente derivable de la posicion actual (`halfmove_clock`), a
diferencia de la repeticion triple que necesita el historial de la linea
(eso vive en repetition.py). is_fifty_move_rule() sigue la convencion de
python-chess: halfmove_clock >= 100 (100 medios-movimientos = 50
movimientos completos sin captura ni movimiento de peon) habilita la regla
- es una condicion de tablas RECLAMABLE en ajedrez real, no automatica;
aca se expone como bandera informativa, no como un forzado automatico de
tablas (ese criterio le corresponde a la logica de partida completa, fuera
del alcance de Fase I)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from search_v1.position import Position


class TerminalStatus(Enum):
    NONE = auto()
    CHECKMATE = auto()
    STALEMATE = auto()
    INSUFFICIENT_MATERIAL = auto()


@dataclass(frozen=True)
class TerminalResult:
    status: TerminalStatus
    is_terminal: bool
    winner: str | None  # "white"/"black" solo si status==CHECKMATE, si no None
    reason: str


def detect_terminal(position: Position) -> TerminalResult:
    board = position.board
    if board.is_checkmate():
        # el lado que ACABA de mover gano - es el opuesto de side_to_move actual
        winner = "black" if position.side_to_move == "white" else "white"
        return TerminalResult(TerminalStatus.CHECKMATE, True, winner, "jaque mate")
    if board.is_stalemate():
        return TerminalResult(TerminalStatus.STALEMATE, True, None, "ahogado")
    if board.is_insufficient_material():
        return TerminalResult(TerminalStatus.INSUFFICIENT_MATERIAL, True, None, "material insuficiente")
    return TerminalResult(TerminalStatus.NONE, False, None, "no terminal")


def is_fifty_move_rule(position: Position) -> bool:
    """True si halfmove_clock >= 100 (regla de 50 movimientos disponible,
    ver docstring del modulo sobre la distincion reclamable/automatica)."""
    return position.halfmove_clock >= 100
