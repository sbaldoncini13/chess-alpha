"""Limites de busqueda de search_v1 (Fase II)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchLimits:
    max_depth: int = 4
    max_nodes: int | None = None
    max_time_ms: float | None = None
    # Frecuencia (en nodos) con la que se consulta el reloj monotonico
    # (time.perf_counter) para el limite de TIEMPO. El limite de NODOS se
    # revisa en CADA nodo (comparacion de enteros, costo despreciable) -
    # solo el chequeo de tiempo (llamada al reloj del sistema) se batchea,
    # que es la parte realmente costosa si se hiciera en cada nodo,
    # documentado explicitamente segun lo pedido ("frecuencia segura y
    # documentada").
    time_check_frequency: int = 1024

    def __post_init__(self):
        if self.max_depth < 0:
            raise ValueError("max_depth no puede ser negativo")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError("max_nodes debe ser >= 1 si se especifica")
        if self.max_time_ms is not None and self.max_time_ms < 0:
            raise ValueError("max_time_ms no puede ser negativo")
        if self.time_check_frequency < 1:
            raise ValueError("time_check_frequency debe ser >= 1")
