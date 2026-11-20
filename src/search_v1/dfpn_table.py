"""Tabla de transposiciones de DFPN/PNS de search_v1 (Fase VI) - NAMESPACE
COMPLETAMENTE SEPARADO de `transposition_table.TranspositionTable` (Fase
III/IV/V, busqueda Alpha-Beta/PVS/quiescence). Reutiliza la MISMA clave
Zobrist (`Position.zobrist_key`) por continuidad de convencion, pero las
entradas tienen semantica propia (proof/disproof number, no score/bound
alfa-beta) y NUNCA se comparten entre las dos tablas - mezclarlas seria
un bug de tipo/semantica, no solo de indice.

REGLA CRITICA (item explicito del pedido): "no reutilizar una prueba bajo
un atacante diferente" - una posicion identica (misma clave Zobrist)
puede estar siendo analizada bajo DOS regimenes distintos de la MISMA
busqueda de Fase VI (¿puede BLANCAS forzar mate desde aqui? vs ¿puede
NEGRAS forzar mate desde aqui?) - son preguntas DIFERENTES aunque la
posicion sea la misma, y sus proof/disproof numbers NO son intercambiables
(node_type tambien se exige identico por la misma razon: el mismo
`position_key` puede visitarse como nodo OR desde una linea y como nodo
AND desde otra si el atacante cambia, aunque en la practica de esta
implementacion `attacker` fijo por busqueda ya lo previene la mayoria de
las veces - se verifica de todas formas, expresamente, sin asumir).

NUNCA se guarda un resultado UNKNOWN (limite alcanzado) como si fuera una
prueba exacta - solo se llama a `store()` con nodos COMPLETAMENTE
resueltos (terminales reales o con pn/dn recalculados de hijos ya
resueltos/expandidos), nunca con un estado parcial cortado a mitad de
expansion por limite de nodos/tiempo (ver dfpn.py)."""

from __future__ import annotations

from dataclasses import dataclass

from search_v1.dfpn_node import NodeType
from search_v1.proof_status import ProofStatus


@dataclass
class DFPNEntry:
    key: int
    attacker: str                       # "white"/"black" - regimen de la prueba
    node_type: NodeType
    pn: int
    dn: int
    expanded: bool
    best_proof_move: str | None         # jugada hacia el hijo mas probante conocido (None si no resuelto/sin hijos)
    children_count: int                 # "children summary" - cantidad de hijos generados (no el subarbol completo)
    generation: int
    terminal_status: ProofStatus | None


class DFPNTable:
    """Direct-mapped (mismo diseno auditable que TranspositionTable de
    Fase III: `key % capacity`, clave completa SIEMPRE comparada antes de
    aceptar un hit - dos posiciones distintas pueden compartir indice sin
    nunca producir un falso hit)."""

    def __init__(self, capacity_entries: int = 65536):
        if capacity_entries < 1:
            raise ValueError("capacity_entries debe ser >= 1")
        self.capacity = capacity_entries
        self._table: list[DFPNEntry | None] = [None] * capacity_entries
        self.generation = 0
        self.probes = 0
        self.hits = 0
        self.collisions = 0            # clave distinta en el mismo indice
        self.regime_mismatches = 0     # misma clave, attacker o node_type distinto - rechazado, NO es una colision de indice
        self.stores = 0
        self.replacements = 0

    def _index(self, key: int) -> int:
        return key % self.capacity

    def new_generation(self) -> None:
        self.generation += 1

    def probe(self, key: int, attacker: str, node_type: NodeType) -> DFPNEntry | None:
        self.probes += 1
        idx = self._index(key)
        entry = self._table[idx]
        if entry is None:
            return None
        if entry.key != key:
            self.collisions += 1
            return None
        if entry.attacker != attacker or entry.node_type != node_type:
            self.regime_mismatches += 1
            return None
        self.hits += 1
        return entry

    def store(
        self, key: int, attacker: str, node_type: NodeType, pn: int, dn: int, expanded: bool,
        best_proof_move: str | None, children_count: int, terminal_status: ProofStatus | None,
    ) -> None:
        new_entry = DFPNEntry(
            key=key, attacker=attacker, node_type=node_type, pn=pn, dn=dn, expanded=expanded,
            best_proof_move=best_proof_move, children_count=children_count,
            generation=self.generation, terminal_status=terminal_status,
        )
        idx = self._index(key)
        existing = self._table[idx]

        if existing is None:
            self._table[idx] = new_entry
            self.stores += 1
            return

        if existing.key == key and existing.attacker == attacker and existing.node_type == node_type:
            # misma posicion+regimen exacto - un resultado RESUELTO (pn==0
            # o dn==0) nunca se pisa por uno no resuelto; entre dos
            # resueltos o dos no resueltos, gana el mas reciente.
            existing_resolved = existing.pn == 0 or existing.dn == 0
            new_resolved = pn == 0 or dn == 0
            if existing_resolved and not new_resolved:
                self.stores += 1
                return
            self._table[idx] = new_entry
            self.replacements += 1
            self.stores += 1
            return

        # bucket ocupado por otra clave/regimen - generacion anterior
        # siempre reemplaza; si no, gana el resuelto sobre el no resuelto,
        # y a igualdad, el mas reciente.
        if existing.generation < self.generation:
            self._table[idx] = new_entry
            self.replacements += 1
        else:
            existing_resolved = existing.pn == 0 or existing.dn == 0
            new_resolved = pn == 0 or dn == 0
            if not (existing_resolved and not new_resolved):
                self._table[idx] = new_entry
                self.replacements += 1
        self.stores += 1

    def occupancy(self) -> float:
        filled = sum(1 for e in self._table if e is not None)
        return filled / self.capacity
