"""Empaquetado de telemetria de quiescence de search_v1 (Fase IV).

No es un dataclass de resultado de busqueda separado (SearchResult de
Fase II ya cubre ese contrato, con `metadata` como diccionario extensible
- mismo patron usado por la TT y el ordering de Fase III) - es la funcion
que junta los contadores `q_*` de SearchContext en un dict listo para
`SearchResult.metadata`, analoga a `iterative_deepening._telemetry_
metadata()` para TT/ordering."""

from __future__ import annotations


def quiescence_telemetry(context) -> dict:
    return {
        "q_nodes": context.q_nodes,
        "q_leaf_nodes": context.q_leaf_nodes,
        "q_terminal_nodes": context.q_terminal_nodes,
        "q_stand_pat_calls": context.q_stand_pat_calls,
        "q_stand_pat_cutoffs": context.q_stand_pat_cutoffs,
        "q_beta_cutoffs": context.q_beta_cutoffs,
        "q_captures_searched": context.q_captures_searched,
        "q_en_passant_searched": context.q_en_passant_searched,
        "q_promotions_searched": context.q_promotions_searched,
        "q_checks_searched": context.q_checks_searched,
        "q_evasions_searched": context.q_evasions_searched,
        "q_max_depth_reached": context.q_max_depth_reached,
        "q_node_limit_stops": context.q_node_limit_stops,
        "q_depth_limit_stops": context.q_depth_limit_stops,
        "q_tt_probes": context.q_tt_probes,
        "q_tt_hits": context.q_tt_hits,
    }
