"""Analisis de defensa obligatoria de search_v1 (Fase VI).

DEFINICION ADOPTADA (documentada explicitamente, alcance EXACTO dentro de
lo que esta capa puede probar): una jugada del lado a mover en `position`
"preserva" el resultado objetivo si, DESPUES de jugarla, el oponente NO
puede forzar jaque mate (probado exhaustivamente via DFPN, `mate_search.py`,
sobre la posicion resultante). Una jugada "pierde" si el oponente SI puede
forzar jaque mate desde la posicion resultante. Si el presupuesto se agota
antes de completar la prueba para una jugada, esa jugada queda "unknown" -
NUNCA se clasifica una jugada como preservadora o perdedora sin una prueba
completa.

Esta es una definicion DELIBERADAMENTE MAS ESTRECHA que "defensa obligatoria"
en sentido amplio de ajedrez (que tambien incluye evitar una perdida
material/posicional demostrada sin llegar a mate forzado) - se elige esta
definicion porque es la UNICA para la que esta fase construyo maquinaria
de PRUEBA EXACTA (DFPN). Documentado explicitamente para no sobre-declarar
alcance: "unique_forced_defense" aca significa especificamente "unica
jugada que evita el mate forzado del oponente", no "unica jugada optima
en sentido evaluativo general"."""

from __future__ import annotations

from dataclasses import dataclass, field

from search_v1.dfpn_limits import DFPNLimits
from search_v1.dfpn_table import DFPNTable
from search_v1.mate_search import find_forced_mate
from search_v1.move import Move
from search_v1.move_generator import generate_legal_moves
from search_v1.position import Position
from search_v1.proof_status import ProofStatus, is_proven


@dataclass
class MoveDefenseClassification:
    move_uci: str
    proof_status: ProofStatus          # resultado de la prueba DFPN sobre la posicion TRAS esta jugada (perspectiva del oponente)
    classification: str                # "preserving" | "losing" | "unknown"
    mate_distance_plies: int | None    # distancia del mate del OPONENTE si classification=="losing"
    refutation_pv: list[str]           # linea forzada del oponente si classification=="losing" (vacia si no)


@dataclass
class ForcedDefenseResult:
    fen: str
    legal_moves_count: int
    preserving_moves: list[str] = field(default_factory=list)
    losing_moves: list[str] = field(default_factory=list)
    unknown_moves: list[str] = field(default_factory=list)
    unique_forced_defense: bool = False
    move_classifications: list[MoveDefenseClassification] = field(default_factory=list)
    limits_reached: bool = False
    nodes_total: int = 0
    elapsed_ms: float = 0.0
    root_position_intact: bool = True


def analyze_forced_defense(
    position: Position, limits: DFPNLimits | None = None, table: DFPNTable | None = None,
) -> ForcedDefenseResult:
    import time
    t0 = time.perf_counter()
    original_fen = position.fen
    limits = limits or DFPNLimits()
    table = table if table is not None else DFPNTable()

    legal_moves = generate_legal_moves(position)
    result = ForcedDefenseResult(fen=original_fen, legal_moves_count=len(legal_moves))

    for lm in sorted(legal_moves, key=lambda m: m.uci):
        position.apply(lm.move)
        try:
            mate_res = find_forced_mate(position, limits=limits, table=table)
        finally:
            position.undo()

        result.nodes_total += mate_res.nodes

        if mate_res.proof_status == ProofStatus.PROVEN_MATE:
            classification = "losing"
            result.losing_moves.append(lm.uci)
        elif mate_res.proof_status in (ProofStatus.PROVEN_NO_MATE, ProofStatus.PROVEN_NO_MATE_WITHIN_BOUND):
            classification = "preserving"
            result.preserving_moves.append(lm.uci)
        else:
            classification = "unknown"
            result.unknown_moves.append(lm.uci)
            result.limits_reached = True

        result.move_classifications.append(MoveDefenseClassification(
            move_uci=lm.uci, proof_status=mate_res.proof_status, classification=classification,
            mate_distance_plies=mate_res.mate_distance_plies if classification == "losing" else None,
            refutation_pv=list(mate_res.principal_variation) if classification == "losing" else [],
        ))

    result.unique_forced_defense = (
        len(result.unknown_moves) == 0 and len(result.preserving_moves) == 1
    )
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    result.root_position_intact = position.fen == original_fen
    return result
