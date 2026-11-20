"""Deteccion de repeticion de posiciones de search_v1.

Trackea la clave de posicion (Position.zobrist_key, ver decision de diseno
en position.py: Polyglot zobrist_hash, sin halfmove clock) a lo largo de
una linea (partida real o linea de busqueda futura). Repeticion en ajedrez
se define por el ESTADO relevante de la posicion (piezas + turno +
enroques + al-paso capturable), NO por el halfmove clock - dos posiciones
con distinto halfmove_clock pero mismo estado de tablero SI cuentan como
la misma posicion a efectos de repeticion. Por eso la clave usada aca
excluye deliberadamente halfmove_clock (ya lo hace zobrist_hash de
Polyglot por diseno)."""

from __future__ import annotations

from collections import Counter

from search_v1.position import Position


def zobrist_key(position: Position) -> int:
    """Alias directo de Position.zobrist_key - expuesto tambien aca porque
    es el modulo natural donde buscar la funcion de clave de repeticion."""
    return position.zobrist_key


class RepetitionTracker:
    """Historial de claves de posicion a lo largo de una linea. No es una
    transposition table (eso es un componente de Fase III, no implementado
    aca) - es solo el conteo de ocurrencias necesario para 3-fold."""

    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()
        self._history: list[int] = []

    def push(self, position: Position) -> None:
        key = zobrist_key(position)
        self._counts[key] += 1
        self._history.append(key)

    def pop(self) -> None:
        if not self._history:
            raise IndexError("RepetitionTracker.pop() sin historial (nada empujado)")
        key = self._history.pop()
        self._counts[key] -= 1
        if self._counts[key] <= 0:
            del self._counts[key]

    def count(self, position: Position) -> int:
        return self._counts.get(zobrist_key(position), 0)

    def is_threefold_repetition(self, position: Position | None = None) -> bool:
        """Si `position` es None, evalua la ULTIMA posicion empujada
        (comportamiento tipico durante una linea de busqueda)."""
        if position is not None:
            key = zobrist_key(position)
        elif self._history:
            key = self._history[-1]
        else:
            return False
        return self._counts.get(key, 0) >= 3

    def __len__(self) -> int:
        return len(self._history)
