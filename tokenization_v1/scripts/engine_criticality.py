"""Extension v2.1 (correccion de auditoria, PASO 27-engine): criticidad
derivada de senales MultiPV de Stockfish. Formula documentada, con
constantes en config (nunca hardcodeadas en silencio) -- ver SCHEMA_V2.md
seccion 'Extension v2.1' para la derivacion completa.

NUNCA reemplaza criticality_score/tactical_or_critical (base, sin motor,
difficulty_criticality.py) -- son columnas y conceptos separados.

Formula (resumen, ver SCHEMA_V2.md para el detalle):
  - si CUALQUIER candidato del top-k es mate (a favor o en contra del que
    mueve): engine_criticality_score = 1.0. Un mate forzado en cualquier
    direccion es, por definicion, la posicion mas critica posible -- no
    tiene sentido matizarlo con el resto de las senales.
  - si no: combinacion lineal, clampeada a [0,1], de:
      * gap_term: diferencia normalizada entre el score (ya unificado
        cp/mate) del candidato rank=1 y el rank=2 (0 si hay <2 candidatos).
      * policy_top1_weight: peso de la soft policy en el candidato rank=1
        (1.0 si solo hay un candidato -- softmax de un unico elemento).
"""


def _gap_term(topk: list, gap_saturation_cp: float):
    if len(topk) < 2:
        return None, 0.0
    gap_cp = topk[0]["eval_unified_cp_equivalent"] - topk[1]["eval_unified_cp_equivalent"]
    term = max(0.0, min(1.0, gap_cp / gap_saturation_cp))
    return float(gap_cp), term


def _policy_top1_weight(policy_rows: list):
    if not policy_rows:
        return None
    by_rank = sorted(policy_rows, key=lambda p: p["rank"])
    return float(by_rank[0]["probability"])


def engine_criticality_score(topk: list, policy_rows: list, config: dict):
    """topk: filas de engine_topk.py para UNA posicion (formato de
    engine_records.py::topk_rows_from_analysis, ya con eval_type/
    eval_unified_cp_equivalent). policy_rows: filas de
    engine_records.py::policy_rows_from_topk para la misma posicion.

    Devuelve (score, any_mate_candidate, gap_cp, policy_top1_weight).
    score es None solo si topk esta vacio (nada que evaluar -- posicion sin
    candidatos, no deberia ocurrir sobre filas con movimiento humano)."""
    if not topk:
        return None, False, None, None

    any_mate = any(c["eval_type"] == "mate" for c in topk)
    if any_mate:
        return 1.0, True, None, _policy_top1_weight(policy_rows)

    gap_cp, gap_term = _gap_term(topk, config["gap_saturation_cp"])
    policy_top1 = _policy_top1_weight(policy_rows)
    policy_term = policy_top1 if policy_top1 is not None else 0.0

    score = config["w_gap"] * gap_term + config["w_policy_concentration"] * policy_term
    score = max(0.0, min(1.0, score))
    return score, False, gap_cp, policy_top1


def engine_tactical_or_critical(score, config: dict):
    if score is None:
        return None
    return score >= config["threshold"]


def engine_criticality_row(row_id: int, topk: list, policy_rows: list, config: dict) -> dict:
    score, any_mate, gap_cp, policy_top1 = engine_criticality_score(topk, policy_rows, config)
    return {
        "row_id": row_id,
        "engine_criticality_score": score,
        "engine_tactical_or_critical": engine_tactical_or_critical(score, config),
        "any_mate_candidate": any_mate,
        "gap_cp": gap_cp,
        "policy_top1_weight": policy_top1,
    }
