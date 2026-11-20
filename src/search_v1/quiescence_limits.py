"""Limites de quiescence de search_v1 (Fase IV).

`max_qdepth` limita las plies de busqueda tactica NORMAL (capturas/
promociones/jaques tacticos opcionales) medida desde el punto donde
alpha_beta entra en quiescence (no desde la raiz de la busqueda
principal). `max_check_extensions` es un presupuesto SEPARADO que solo se
consume cuando la posicion sigue en jaque despues de agotar max_qdepth
(nunca se hace stand-pat en jaque, asi que hay que poder seguir buscando
evasiones mas alla del limite normal) - al agotarse, la busqueda se
cancela de forma limpia (SearchAborted) exactamente igual que un limite
de nodos/tiempo, en vez de forzar un valor incorrecto. `max_qnodes` es un
techo GLOBAL (compartido por toda la llamada a iterative_deepening_search,
no por cada sub-arbol de quiescence individual) - previene explosiones
combinatorias reales sin necesitar podas silenciosas no documentadas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuiescenceLimits:
    max_qdepth: int = 8
    max_qnodes: int | None = None
    max_check_extensions: int = 6
    allow_tactical_checks: bool = True
    # profundidad hasta la que se generan jaques tacticos no capturantes
    # (mas alla de esto, en posiciones NO en jaque, solo capturas/
    # promociones - limita el costo de detectar "da jaque" en cascada)
    tactical_checks_max_qdepth: int = 3

    def __post_init__(self):
        if self.max_qdepth < 0:
            raise ValueError("max_qdepth no puede ser negativo")
        if self.max_qnodes is not None and self.max_qnodes < 1:
            raise ValueError("max_qnodes debe ser >= 1 si se especifica")
        if self.max_check_extensions < 0:
            raise ValueError("max_check_extensions no puede ser negativo")
        if self.tactical_checks_max_qdepth < 0:
            raise ValueError("tactical_checks_max_qdepth no puede ser negativo")
