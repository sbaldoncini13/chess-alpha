"""Representacion de posicion de search_v1.

DECISION DE DISENO (documentada explicitamente, tal como exige el pedido de
Fase I): la capa de representacion de tablero es `python-chess`
(`chess.Board`), NO una reimplementacion propia (bitboards, etc.). Esto ya
estaba establecido en el diseno tecnico aprobado
(reports/search_tree_v1/SEARCH_TREE_V1_TECHNICAL_DESIGN.md, seccion 2.1:
"Dependencia: python-chess, ya usado en todo el pipeline de datos, probado
a fondo") y es consistente con el resto del proyecto: las 39.961 posiciones
de SEARCH TREE DATASET V1.1 fueron generadas, validadas y hasheadas con
python-chess (chess_alpha/search_tree_dataset/identity.py). Reimplementar
generacion de movimientos desde cero duplicaria una libreria ya validada
exhaustivamente sobre este mismo dataset, con mucho mas riesgo de bugs
sutiles de legalidad. python-chess NO se usa aqui como "oraculo de
comparacion en tests" solamente - es la base real de la representacion,
como permite explicitamente el pedido de Fase I.

Clave de posicion (Zobrist/Polyglot): se reutiliza `chess.polyglot.
zobrist_hash`, EXACTAMENTE la misma funcion usada como `position_id` en
identity.py para las 39.961 posiciones del dataset (continuidad de
convencion, no una eleccion nueva). Esta funcion combina disposicion de
piezas + lado al turno + derechos de enroque + al paso (solo cuando es
realmente capturable) - por diseno de Polyglot, NO incluye halfmove clock
ni fullmove number, que es exactamente el requisito pedido para la clave de
repeticion (no mezclar el halfmove clock)."""

from __future__ import annotations

import chess
import chess.polyglot

from search_v1.exceptions import IllegalMoveError, InvalidFENError, UndoStackEmptyError
from search_v1.move import Move

STARTING_FEN = chess.STARTING_FEN


def _validate_fen(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise InvalidFENError(fen, f"error de parseo: {exc}") from exc

    status = board.status()
    if status != chess.STATUS_VALID:
        problems = [
            name for name in dir(chess)
            if name.startswith("STATUS_") and name != "STATUS_VALID"
            and getattr(chess, name) & status
        ]
        raise InvalidFENError(fen, f"tablero semanticamente invalido: {problems}")
    return board


class Position:
    """Wrapper fino sobre chess.Board. Expone la API minima que search_v1
    necesita, sin filtrar el objeto interno como unica via (aunque `.board`
    esta disponible para quien lo necesite, ver decision de diseno arriba)."""

    __slots__ = ("_board", "_push_count")

    def __init__(self, fen: str | None = None):
        self._board = _validate_fen(fen) if fen is not None else chess.Board()
        self._push_count = 0

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        return cls(fen)

    @property
    def board(self) -> chess.Board:
        """Acceso directo al chess.Board interno - ver decision de diseno
        en el docstring del modulo. No copia; mutarlo por fuera de
        apply()/undo() rompe la contabilidad de _push_count."""
        return self._board

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def side_to_move(self) -> str:
        return "white" if self._board.turn == chess.WHITE else "black"

    @property
    def castling_rights(self) -> str:
        """Substring de disponibilidad de enroque del FEN (ej. 'KQkq', '-')."""
        part = self._board.fen().split(" ")[2]
        return part

    @property
    def en_passant_square(self) -> str | None:
        part = self._board.fen().split(" ")[3]
        return None if part == "-" else part

    @property
    def halfmove_clock(self) -> int:
        return self._board.halfmove_clock

    @property
    def fullmove_number(self) -> int:
        return self._board.fullmove_number

    @property
    def zobrist_key(self) -> int:
        """Clave de posicion para repeticion/futura TT - NO incluye
        halfmove clock (ver docstring del modulo)."""
        return chess.polyglot.zobrist_hash(self._board)

    @property
    def position_id(self) -> str:
        """Mismo formato que `position_id` en identity.py (hex de 16
        caracteres) - permite cruzar directamente con SEARCH TREE DATASET
        V1.1 sin conversion adicional."""
        return format(self.zobrist_key, "016x")

    @property
    def is_check(self) -> bool:
        return self._board.is_check()

    def copy(self) -> "Position":
        new = Position.__new__(Position)
        new._board = self._board.copy()
        new._push_count = 0
        return new

    def legal_move_ucis(self) -> list[str]:
        return [m.uci() for m in self._board.legal_moves]

    def is_legal(self, move: Move) -> bool:
        return move.raw in self._board.legal_moves

    def apply(self, move: Move) -> None:
        """Aplica un movimiento LEGAL. Lanza IllegalMoveError si no lo es -
        nunca aplica un movimiento pseudo-legal en silencio."""
        if not self.is_legal(move):
            raise IllegalMoveError(move.uci, self.fen)
        self._board.push(move.raw)
        self._push_count += 1

    def undo(self) -> None:
        """Deshace el ultimo movimiento aplicado via apply(). Restaura el
        estado EXACTO previo (delegado a chess.Board.pop(), que restaura
        halfmove_clock/fullmove_number/derechos de enroque/al-paso
        exactamente, no solo la disposicion de piezas)."""
        if self._push_count == 0:
            raise UndoStackEmptyError()
        self._board.pop()
        self._push_count -= 1

    def apply_null(self) -> None:
        """Aplica un 'movimiento nulo' (pasa el turno, no mueve ninguna
        pieza, limpia al-paso) - exclusivo de null move pruning (Fase V).
        NUNCA es una jugada legal real - no pasa por is_legal(). El
        LLAMADOR es responsable de nunca invocarlo estando en jaque (esa
        es una guarda de la tecnica de null move pruning, no de esta
        funcion). Reusa el MISMO stack/contador que apply() - se deshace
        con el mismo undo(), sin metodo separado, porque chess.Board.pop()
        no distingue el tipo de la ultima jugada empujada."""
        self._board.push(chess.Move.null())
        self._push_count += 1

    def gives_check(self, move: Move) -> bool:
        return self._board.gives_check(move.raw)

    def is_capture(self, move: Move) -> bool:
        return self._board.is_capture(move.raw)

    def is_en_passant(self, move: Move) -> bool:
        return self._board.is_en_passant(move.raw)

    def is_castling(self, move: Move) -> bool:
        return self._board.is_castling(move.raw)

    def san(self, move: Move) -> str:
        return self._board.san(move.raw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        # Igualdad ESTRICTA por FEN completo (incluye halfmove/fullmove) -
        # usada por los tests de apply/undo ("conservacion exacta del
        # estado"), distinta de zobrist_key (que deliberadamente ignora
        # halfmove/fullmove para la clave de repeticion).
        return self.fen == other.fen

    def __repr__(self) -> str:
        return f"Position({self.fen!r})"
