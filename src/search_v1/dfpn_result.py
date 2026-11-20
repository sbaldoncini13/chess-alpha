"""Resultado estructurado de una prueba de DFPN/PNS de search_v1 (Fase VI)
- mismo patron que `search_result.SearchResult` (Fase II): `metadata` como
diccionario extensible para telemetria, en vez de seguir agregando campos
al dataclass principal."""

from __future__ import annotations

from dataclasses import dataclass, field

from search_v1.proof_status import ProofStatus


@dataclass
class DFPNResult:
    proof_status: ProofStatus
    attacker: str                      # "white"/"black" - lado que la prueba intenta demostrar que mata
    principal_variation: list[str]     # PV UCI desde la raiz hasta el jaque mate (vacia si no PROVEN_MATE)
    mate_distance_plies: int | None    # plies hasta el mate (solo si proof_status==PROVEN_MATE)
    nodes: int
    root_pn: int
    root_dn: int
    elapsed_ms: float
    root_position_intact: bool
    metadata: dict = field(default_factory=dict)
