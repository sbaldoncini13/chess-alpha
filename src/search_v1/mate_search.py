"""Busqueda EXACTA de mate forzado de search_v1 (Fase VI) - capa publica
delgada sobre `dfpn.search_dfpn`. Documentada por separado (modulo propio,
tal como pide la arquitectura sugerida) porque conceptualmente es una
pregunta distinta de "resolver la posicion completa via DFPN": "¿el lado
a mover fuerza jaque mate?" - sin heuristicas, sin podas, exhaustiva
dentro del presupuesto dado."""

from __future__ import annotations

from search_v1.dfpn import search_dfpn
from search_v1.dfpn_limits import DFPNLimits
from search_v1.dfpn_result import DFPNResult
from search_v1.dfpn_table import DFPNTable
from search_v1.position import Position


def find_forced_mate(
    position: Position, limits: DFPNLimits | None = None, table: DFPNTable | None = None,
    game_repetition=None,
) -> DFPNResult:
    """Prueba si `position.side_to_move` fuerza jaque mate. Envoltorio
    fino de `search_dfpn` - ver ese modulo para el algoritmo completo."""
    return search_dfpn(position, limits=limits, table=table, game_repetition=game_repetition)
