"""Validador de principal variation de search_v1 (Fase II).

Reproduce la PV completa desde la posicion raiz (SOBRE UNA COPIA, nunca
muta la posicion recibida) y verifica: cada movimiento es legal en su
turno, ningun movimiento aparece despues de una posicion terminal, el
primer movimiento coincide con best_move, longitud compatible con
completed_depth.

LONGITUD DE PV CON QUIESCENCE (Fase IV): cuando la busqueda usa
quiescence, la PV real puede extenderse mas alla de `completed_depth` -
la extension tactica del horizonte (capturas/jaques/evasiones, incluidas
extensiones de jaque) agrega plies legitimos a la PV que NO representan
un error. El limite de longitud se relaja por
`metadata['q_max_depth_reached']` (maximo qdepth_used observado en
CUALQUIER excursion de quiescence de la busqueda, 0 si no hubo
quiescence) - BUG REAL de Fase IV encontrado en la validacion contra el
dataset: el chequeo original (`len(pv) > completed_depth`) marcaba como
"problema" el 100% de una muestra real de posiciones de quiescence (187
falsos positivos, 0 con movimientos ilegales/posicion mutada reales) -
la PV extendida por quiescence es el comportamiento CORRECTO y
esperado, no una anomalia."""

from __future__ import annotations

from search_v1.move import Move
from search_v1.move_generator import is_legal_uci
from search_v1.position import Position
from search_v1.search_result import SearchResult
from search_v1.terminal import detect_terminal


def validate_principal_variation(position: Position, result: SearchResult) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacia si la PV es
    valida)."""
    problems = []
    pv = result.principal_variation

    if result.best_move is not None and pv and pv[0] != result.best_move:
        problems.append(f"PV[0]={pv[0]!r} no coincide con best_move={result.best_move!r}")

    q_max_depth_reached = result.metadata.get("q_max_depth_reached", 0) or 0
    max_allowed_length = result.completed_depth + q_max_depth_reached
    if len(pv) > max_allowed_length:
        problems.append(
            f"PV mas larga ({len(pv)}) que completed_depth ({result.completed_depth}) "
            f"+ q_max_depth_reached ({q_max_depth_reached}) = {max_allowed_length}"
        )

    working = position.copy()
    for i, uci in enumerate(pv):
        terminal = detect_terminal(working)
        if terminal.is_terminal:
            problems.append(f"PV contiene un movimiento ({uci!r}) despues de una posicion terminal en el indice {i}")
            break
        if not is_legal_uci(working, uci):
            problems.append(f"PV[{i}]={uci!r} no es legal en la posicion alcanzada")
            break
        working.apply(Move.from_uci(uci))

    return problems
