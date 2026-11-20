"""Nodo del arbol de Proof-Number Search de search_v1 (Fase VI).

DECISION DE DISENO (documentada explicitamente): se implementa
Proof-Number Search (Allis 1994) con arbol EXPLICITO en memoria (no DFPN
clasico con pasaje de umbrales/thresholds "delayed") - el pedido permite
explicitamente "Proof-Number Search o DFPN... variante equivalente
correctamente documentada". Se elige PNS con arbol explicito por ser
significativamente mas simple de verificar (sin la logica de umbrales
duales de df-pn, una fuente conocida de bugs sutiles en implementaciones
reales) - a la escala de esta fase (mates en pocos plies, defensa
obligatoria en profundidad acotada, presupuestos de nodos moderados) la
diferencia de eficiencia frente a df-pn clasico no es relevante, y la
correccion demostrable pesa mas. La deteccion de transposiciones/ciclos
se resuelve via `dfpn_table.py` (memoizacion de posiciones YA resueltas,
nunca fusion de nodos en un DAG con multiples padres - evita la clase de
bug de propagacion ascendente con padres multiples).

SEMANTICA OR/AND: el "atacante" es el lado a mover en la RAIZ de la
prueba (fijo durante toda la busqueda) - se intenta demostrar que el
atacante fuerza jaque mate. Nodo OR = atacante a mover (elige CUALQUIER
jugada que lleve a un nodo probado - basta UNA). Nodo AND = defensor a
mover (debe sobrevivir a TODAS las respuestas - alcanza con UNA que
sobreviva para refutar). `pn`=0 significa "probado" (el atacante fuerza
mate desde aqui); `dn`=0 significa "refutado" (el atacante NO fuerza mate
desde aqui, bajo el regimen de esta busqueda)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from search_v1.proof_status import ProofStatus

# Infinito representado como entero grande (no float) - evita aritmetica
# de punto flotante en sumas/comparaciones de proof/disproof numbers,
# que deben ser exactas. Suficientemente grande para nunca confundirse
# con una suma real de proof numbers en las profundidades de esta fase.
INFINITY_PN = 1_000_000_000


class NodeType(Enum):
    OR = auto()   # atacante a mover
    AND = auto()  # defensor a mover


@dataclass
class DFPNNode:
    node_type: NodeType
    move_uci: str | None = None          # jugada del PADRE que lleva a este nodo (None en la raiz)
    position_key: int | None = None      # se completa recien al EXPANDIR el nodo (ver dfpn.py::_expand)
    pn: int = 1
    dn: int = 1
    expanded: bool = False
    children: list["DFPNNode"] = field(default_factory=list)
    terminal_status: ProofStatus | None = None  # set solo en hojas terminales reales (no en UNKNOWN)
    # False SOLO para un ciclo interno del CAMINO de busqueda o un corte
    # de profundidad (ver dfpn.py::_classify_terminal/_expand) - nunca se
    # almacena un nodo asi en la tabla de transposiciones (bug real
    # corregido: esa clasificacion depende del camino, no solo de la
    # posicion), y se propaga a cualquier ancestro cuyo pn/dn dependa de
    # el (ver dfpn.py::_recompute).
    cacheable: bool = True
    # True cuando este nodo se resolvio REUTILIZANDO una entrada ya
    # existente en la tabla (ver dfpn.py::_expand) - un nodo asi NUNCA debe
    # volver a guardarse en la tabla (bug real corregido: volver a
    # guardarlo podia PISAR, por politica de reemplazo entre dos entradas
    # "resueltas", una entrada mas informativa con `best_proof_move` real
    # por una copia con `best_proof_move=None` - ver test_forced_defense.py).
    from_tt_reuse: bool = False

    @property
    def is_proven(self) -> bool:
        return self.pn == 0

    @property
    def is_disproven(self) -> bool:
        return self.dn == 0

    @property
    def is_resolved(self) -> bool:
        return self.pn == 0 or self.dn == 0

    def opposite_type(self) -> NodeType:
        return NodeType.AND if self.node_type == NodeType.OR else NodeType.OR
