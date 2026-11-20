"""Fusion de candidatos Epoch4 + Sequence Policy de search_v1 (Fase VIII,
modo D = epoch4_sequence). Combina las dos senales de ordenamiento SOLO en
espacio de movimientos UCI - NUNCA mezclando los indices numericos de los
dos vocabularios (Epoch4 usa 1.868 indices fijos; Sequence Policy usa un
espacio de action_id completamente distinto, 20.483 - combinarlos por
indice seria un bug silencioso, regla explicita ya documentada en
reports/search_tree_v1/SEARCH_TREE_V1_TECHNICAL_DESIGN.md seccion 4).

REGLA CENTRAL (identica a los dos proveedores individuales): la fusion
SOLO reordena - jamas agrega, quita o cambia el conjunto de jugadas
exploradas por PVS. `reorder_with_fusion` es una funcion PURA con el
mismo contrato de permutacion exacta que `reorder_with_epoch4`/
`reorder_with_sequence_policy`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = False
    rrf_k: int = 60  # default estandar de literatura IR (Cormack et al. 2009) - NO ajustado contra el dev set de este proyecto, ver plan de Fase VIII "fuera de alcance"

    def __post_init__(self):
        if self.rrf_k < 1:
            raise ValueError("rrf_k debe ser >= 1")


def _ranks_by_uci(probs: dict) -> dict:
    """Rank 1-indexado dentro de la propia fuente, descendente por
    probabilidad. Por que RANK y no probabilidad cruda: las dos fuentes
    viven en escalas no comparables (Epoch4: softmax sobre <=1.868 vias;
    Sequence Policy: softmax legal-only sobre unas pocas vias por
    posicion) - promediar probabilidades crudas compararia numeros que no
    significan lo mismo. El rank dentro de la propia fuente es
    invariante de escala."""
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return {uci: i + 1 for i, (uci, _prob) in enumerate(ordered)}


def reorder_with_fusion(moves: list, hash_move_uci: str | None,
                         epoch4_probs: dict, sequence_probs: dict,
                         fusion_config: FusionConfig) -> list:
    """Funcion PURA - mismo contrato de permutacion exacta que los
    proveedores individuales. "Cubiertas" = UNION de los UCIs presentes en
    CUALQUIERA de las dos fuentes (combinacion siempre en espacio UCI).
    Formula RRF (Reciprocal Rank Fusion, prior art ya documentado en este
    proyecto - ver fusion.py del bundle de Sequence Policy y
    docs/PROJECT_STATE.md seccion B): `fused_score(uci) =
    1/(k+rank_epoch4[uci]) + 1/(k+rank_sequence[uci])`, donde un rank
    ausente en una fuente contribuye 0 (NUNCA excluye la jugada de la
    union - misma regla que el `reciprocal_rank_fusion` original)."""
    if not moves:
        return []

    pinned = None
    rest = moves
    if moves[0].uci == hash_move_uci:
        pinned = moves[0]
        rest = moves[1:]

    rank_epoch4 = _ranks_by_uci(epoch4_probs)
    rank_sequence = _ranks_by_uci(sequence_probs)
    k = fusion_config.rrf_k

    def fused_score(uci: str) -> float:
        score = 0.0
        if uci in rank_epoch4:
            score += 1.0 / (k + rank_epoch4[uci])
        if uci in rank_sequence:
            score += 1.0 / (k + rank_sequence[uci])
        return score

    covered_ucis = set(epoch4_probs) | set(sequence_probs)
    covered = [lm for lm in rest if lm.uci in covered_ucis]
    uncovered = [lm for lm in rest if lm.uci not in covered_ucis]
    covered.sort(key=lambda lm: fused_score(lm.uci), reverse=True)

    result = ([pinned] if pinned is not None else []) + covered + uncovered

    assert len(result) == len(moves), "reorder_with_fusion no debe cambiar la cantidad de jugadas"
    assert {lm.uci for lm in result} == {lm.uci for lm in moves}, "reorder_with_fusion no debe cambiar el conjunto de jugadas"

    return result
