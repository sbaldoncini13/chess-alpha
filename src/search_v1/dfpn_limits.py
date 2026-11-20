"""Limites de DFPN/PNS de search_v1 (Fase VI).

`max_depth_plies` es el UNICO limite que degrada la prueba a
PROVEN_NO_MATE_WITHIN_BOUND en vez de PROVEN_NO_MATE (una prueba
exhaustiva real, sin cota de profundidad, solo puede terminar en
PROVEN_MATE o PROVEN_NO_MATE - ver dfpn.py). `max_nodes`/`max_time_ms`
agotados SIEMPRE producen UNKNOWN_NODE_LIMIT/UNKNOWN_TIME_LIMIT, nunca una
prueba - agotar presupuesto no es evidencia de nada."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFPNLimits:
    max_nodes: int | None = 50_000
    max_time_ms: float | None = None
    # cota EXPLICITA de plies desde la raiz de la prueba de DFPN - si se
    # alcanza sin encontrar mate, el resultado es PROVEN_NO_MATE_WITHIN_BOUND
    # (no PROVEN_NO_MATE: pudiera existir un mate mas alla de la cota).
    # None = sin cota de profundidad (solo nodos/tiempo limitan la prueba).
    max_depth_plies: int | None = None
    time_check_frequency: int = 256

    def __post_init__(self):
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("max_nodes debe ser >= 1 si se especifica")
        if self.max_time_ms is not None and self.max_time_ms < 0:
            raise ValueError("max_time_ms no puede ser negativo")
        if self.max_depth_plies is not None and self.max_depth_plies < 1:
            raise ValueError("max_depth_plies debe ser >= 1 si se especifica")
        if self.time_check_frequency < 1:
            raise ValueError("time_check_frequency debe ser >= 1")
