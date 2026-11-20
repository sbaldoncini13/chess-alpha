"""Generacion de movimientos LEGALES (no pseudo-legales) de search_v1.

`chess.Board.legal_moves` ya excluye pseudo-legales que dejan al propio
rey en jaque (clavadas, no resolver un jaque existente, etc.) - es
exactamente la distincion pedida ("movimientos legales, no solo
pseudolegales"). Este modulo no reimplementa esa logica, la envuelve y le
agrega las banderas (captura/promocion/enroque/al-paso/jaque) con el mismo
esquema de campos ya usado en position_moves.parquet y en los shards de
Nivel C de SEARCH TREE DATASET V1.1 (is_capture/is_promotion/is_castling/
is_en_passant/gives_check) - deliberado, para que la validacion contra el
dataset (Fase I, seccion "validacion contra dataset") compare campo a
campo sin traducir esquemas."""

from __future__ import annotations

from dataclasses import dataclass

from search_v1.move import Move
from search_v1.position import Position


@dataclass(frozen=True)
class LegalMove:
    move: Move
    is_capture: bool
    is_promotion: bool
    is_castling: bool
    is_en_passant: bool
    gives_check: bool

    @property
    def uci(self) -> str:
        return self.move.uci


def generate_legal_moves(position: Position) -> list[LegalMove]:
    result = []
    for raw in position.board.legal_moves:
        move = Move(raw)
        result.append(LegalMove(
            move=move,
            is_capture=position.is_capture(move),
            is_promotion=move.is_promotion,
            is_castling=position.is_castling(move),
            is_en_passant=position.is_en_passant(move),
            gives_check=position.gives_check(move),
        ))
    return result


def legal_move_count(position: Position) -> int:
    return position.board.legal_moves.count()


def is_legal_uci(position: Position, uci: str) -> bool:
    try:
        move = Move.from_uci(uci)
    except ValueError:
        return False
    return position.is_legal(move)
