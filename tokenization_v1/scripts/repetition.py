"""Repeticion de posiciones dentro de UNA partida (PASO 20). Distinto de la
deduplicacion global de position_dedup.py: aca interesa saber, mientras se
recorre una partida especifica, si esta es la 1ra/2da/3ra+ vez que aparece
esta posicion EN ESA PARTIDA (informacion necesaria para la regla de triple
repeticion, que una FEN aislada no contiene).

Usa el mismo position_id (zobrist) que position_dedup.py para que "misma
posicion" signifique lo mismo en ambos lugares.
"""
from position_dedup import position_id


class InGameRepetitionTracker:
    def __init__(self):
        self._seen = {}

    def observe(self, board) -> int:
        """Registra la posicion actual y devuelve cuantas veces (contando
        esta) se vio dentro de esta partida."""
        pid = position_id(board)
        count = self._seen.get(pid, 0) + 1
        self._seen[pid] = count
        return count
