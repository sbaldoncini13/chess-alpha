"""Interfaz basica de resultados de search_v1 (item 13 del alcance de Fase
I). PositionAnalysis es el contrato de salida de esta fase: junta
generacion de movimientos + terminales + evaluacion estatica + clave de
repeticion en un unico objeto. Los campos de busqueda futura (best_move,
score, nodes, depth, pv) existen ya en el dataclass, siempre en None en
esta fase - se documentan como reservados en vez de omitirse, para que el
contrato de la interfaz no tenga que romperse (agregar campos nuevos) en
Fase II, solo dejar de estar en None."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from search_v1.move_generator import LegalMove, generate_legal_moves
from search_v1.position import Position
from search_v1.static_evaluation import evaluate_static_white_pov
from search_v1.terminal import TerminalResult, detect_terminal, is_fifty_move_rule


@dataclass
class PositionAnalysis:
    fen: str
    position_id: str
    side_to_move: str
    zobrist_key: int
    is_check: bool
    legal_moves: list[LegalMove]
    legal_move_count: int
    terminal: TerminalResult
    fifty_move_rule_available: bool
    static_evaluation_white_pov_cp: int
    elapsed_s: float

    # Reservado para Fase II en adelante - siempre None en Fase I.
    best_move: str | None = field(default=None)
    score: float | None = field(default=None)
    nodes: int | None = field(default=None)
    depth: int | None = field(default=None)
    pv: list[str] | None = field(default=None)


def analyze_position(position: Position) -> PositionAnalysis:
    start = time.perf_counter()
    legal_moves = generate_legal_moves(position)
    terminal = detect_terminal(position)
    static_eval = evaluate_static_white_pov(position)
    fifty = is_fifty_move_rule(position)
    elapsed = time.perf_counter() - start
    return PositionAnalysis(
        fen=position.fen,
        position_id=position.position_id,
        side_to_move=position.side_to_move,
        zobrist_key=position.zobrist_key,
        is_check=position.is_check,
        legal_moves=legal_moves,
        legal_move_count=len(legal_moves),
        terminal=terminal,
        fifty_move_rule_available=fifty,
        static_evaluation_white_pov_cp=static_eval,
        elapsed_s=elapsed,
    )
