"""Movimiento de search_v1 - wrapper fino sobre chess.Move.

Las propiedades que dependen del TABLERO (is_capture, is_castling,
is_en_passant, gives_check) viven en Position, no aca - un chess.Move por
si solo (from_square/to_square/promotion) no alcanza para saberlas: por
ejemplo si una casilla esta ocupada solo se puede responder mirando el
tablero. Mantener esa dependencia explicita evita el bug clasico de
"Move que miente" (una instancia de Move construida para un tablero usada
por error contra otro)."""

from __future__ import annotations

import chess

PROMOTION_PIECES = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
}
PROMOTION_LETTERS = {v: k for k, v in PROMOTION_PIECES.items()}


class Move:
    """Wrapper inmutable sobre chess.Move. `.raw` expone el objeto
    chess.Move interno para quien necesite pasarlo de vuelta a
    python-chess (ej. Position.apply)."""

    __slots__ = ("raw",)

    def __init__(self, raw: chess.Move):
        self.raw = raw

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        try:
            return cls(chess.Move.from_uci(uci))
        except ValueError as exc:
            raise ValueError(f"UCI invalido {uci!r}: {exc}") from exc

    @property
    def uci(self) -> str:
        return self.raw.uci()

    @property
    def from_square(self) -> str:
        return chess.square_name(self.raw.from_square)

    @property
    def to_square(self) -> str:
        return chess.square_name(self.raw.to_square)

    @property
    def is_promotion(self) -> bool:
        return self.raw.promotion is not None

    @property
    def promotion_piece(self) -> str | None:
        if self.raw.promotion is None:
            return None
        return PROMOTION_LETTERS[self.raw.promotion]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return self.raw == other.raw

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return f"Move({self.uci!r})"
