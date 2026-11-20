"""Generacion de movimientos tacticos de search_v1 (Fase IV).

Cuando la posicion esta EN JAQUE, `generate_tactical_moves` devuelve
TODAS las jugadas legales (no solo capturas) - por definicion,
`generate_legal_moves` de una posicion en jaque ya son exclusivamente
evasiones (python-chess excluye cualquier pseudo-legal que deje al rey en
jaque), asi que "todas las evasiones legales" = simplemente la lista
completa de jugadas legales, sin necesidad de logica especial de
generacion. Esto es deliberado: reduce el riesgo de "evasion legal
omitida" (criterio de FAIL explicito) a cero, porque no se reimplementa
la deteccion de evasiones - se reutiliza integramente move_generator.py,
ya validado en Fase I.

Cuando NO esta en jaque, el subconjunto tactico es: capturas (incluye al
paso, ya reflejado en LegalMove.is_capture) + promociones (con o sin
captura) + opcionalmente jaques no capturantes (`allow_tactical_checks`,
acotado por `tactical_checks_max_qdepth` - ver quiescence_limits.py)."""

from __future__ import annotations

from search_v1.move_generator import LegalMove, generate_legal_moves
from search_v1.position import Position


def generate_tactical_moves(position: Position, allow_tactical_checks: bool = False) -> list[LegalMove]:
    all_moves = generate_legal_moves(position)

    if position.is_check:
        return all_moves  # evasiones completas, sin excepcion

    tactical = [lm for lm in all_moves if lm.is_capture or lm.is_promotion]

    if allow_tactical_checks:
        tactical_ucis = {lm.uci for lm in tactical}
        tactical += [lm for lm in all_moves if lm.gives_check and lm.uci not in tactical_ucis]

    return tactical
