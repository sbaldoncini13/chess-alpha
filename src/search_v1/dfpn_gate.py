"""Gate configurable de integracion PVS + DFPN de search_v1 (Fase VI) -
"DFPN no debe ejecutarse indiscriminadamente en cada nodo" (item explicito
del pedido). Desactivado por defecto (`enabled=False`) - ver `pvs.py` para
donde se consulta exactamente."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFPNGateConfig:
    enabled: bool = False
    trigger_on_check: bool = True   # solo se activa si el nodo esta en jaque (jaques forzados disponibles - la condicion mas barata y de mayor valor para detectar mate cercano)
    max_pvs_depth_to_trigger: int = 6   # solo si la profundidad NORMAL restante del nodo PVS es <= esto - evita gastar presupuesto de DFPN en nodos muy superficiales de un arbol grande
    dfpn_max_nodes: int = 2000          # presupuesto ACOTADO por invocacion - nunca ilimitado
    dfpn_max_depth_plies: int | None = 8

    def __post_init__(self):
        if self.max_pvs_depth_to_trigger < 1:
            raise ValueError("max_pvs_depth_to_trigger debe ser >= 1")
        if self.dfpn_max_nodes < 1:
            raise ValueError("dfpn_max_nodes debe ser >= 1")
        if self.dfpn_max_depth_plies is not None and self.dfpn_max_depth_plies < 1:
            raise ValueError("dfpn_max_depth_plies debe ser >= 1 si se especifica")
