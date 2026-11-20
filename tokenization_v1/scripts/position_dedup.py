"""Deduplicacion de posiciones a nivel dataset (PASO 4): Zobrist hash via
chess.polyglot (estandar, publico, incluye turno/enroques/al-paso -- a
diferencia de position_hash() de src/chess_alpha/board_features.py que
excluye enroques/al-paso a proposito para agrupar transposiciones "visuales";
aqui se quiere identidad FUNCIONAL exacta de la posicion, por eso zobrist).

No se elimina ninguna posicion duplicada: solo se cuenta la frecuencia.
"""
import chess.polyglot


def position_id(board) -> int:
    """uint64. Identidad funcional exacta de la posicion (piezas + turno +
    enroques + casilla al paso), independiente del camino de jugadas."""
    return chess.polyglot.zobrist_hash(board)


class GlobalOccurrenceCounter:
    """Pasada A: acumula {position_id: occurrence_count} sobre TODO el
    dataset antes de escribir ninguna fila (necesario para que
    position_occurrence_count sea el conteo FINAL, no uno parcial
    creciente segun PASO 4)."""

    def __init__(self):
        self._counts = {}

    def add(self, pid: int):
        self._counts[pid] = self._counts.get(pid, 0) + 1

    def count_of(self, pid: int) -> int:
        return self._counts.get(pid, 0)

    def __len__(self):
        return len(self._counts)
