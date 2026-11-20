"""Telemetria basica de search_v1 (Fase II) - funciones puras, sin
efectos secundarios ni logging propio (quien llame decide que hacer con
los numeros). No se inventa precision temporal mayor que la realmente
medida: nodes_per_second es 0.0 (no NaN/Inf) cuando elapsed_ms es 0."""

from __future__ import annotations


def nodes_per_second(nodes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return nodes / (elapsed_ms / 1000.0)


def branching_factor(nodes: int, depth: int) -> float | None:
    """Factor de ramificacion efectivo aproximado (nodes^(1/depth)) - solo
    informativo, None si depth<=0 (no aplica) o nodes<=0."""
    if depth <= 0 or nodes <= 0:
        return None
    return nodes ** (1.0 / depth)
