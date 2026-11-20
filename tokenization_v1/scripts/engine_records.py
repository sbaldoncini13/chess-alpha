"""Extension v2.1 (correccion de auditoria, puntos 21-26): construye las
filas persistibles de engine_eval / engine_topk / engine_policy_target a
partir de la salida YA EXISTENTE de StockfishEnricher.evaluate_human_move y
softmax_policy_from_topk (stockfish_enrichment.py, sin modificar).

Antes de esta extension, el pipeline de demo solo guardaba
`topk_size`/`policy_target_size` (enteros) -- este modulo produce las filas
reales, con la distincion explicita eval_type='cp'|'mate' que evita mezclar
un mate con un centipawn score de forma ambigua (ver SCHEMA_V2.md,
'Extension v2.1').
"""
from stockfish_enrichment import CP_MATE_SCORE, softmax_policy_from_topk


def _decompose_eval(score_cp: int, is_mate: bool, mate_in):
    """score_cp ya viene del score_to_cp() unificado existente (mate
    mapeado via CP_MATE_SCORE). Devuelve (eval_type, eval_cp, mate_in,
    unified_cp_equivalent) sin perder ninguna de las dos representaciones."""
    if is_mate:
        return "mate", None, int(mate_in), int(score_cp)
    return "cp", float(score_cp), None, int(score_cp)


def topk_rows_from_analysis(row_id: int, analysis: dict, depth: int) -> list:
    """PASO 23. Una fila por candidato (row_id, rank). `analysis` es el dict
    devuelto por StockfishEnricher.analyse_position/evaluate_human_move
    (contiene analysis['topk'] con rank/move_id/move_uci/score_cp/is_mate/mate_in)."""
    rows = []
    for c in analysis["topk"]:
        eval_type, eval_cp, mate_in, unified = _decompose_eval(c["score_cp"], c["is_mate"], c["mate_in"])
        rows.append({
            "row_id": row_id, "rank": c["rank"], "move_id": c["move_id"], "move_uci": c["move_uci"],
            "eval_type": eval_type, "eval_cp": eval_cp, "mate_in": mate_in,
            "eval_unified_cp_equivalent": unified, "depth": depth,
        })
    return rows


def policy_rows_from_topk(row_id: int, topk: list, temperature_cp: float) -> list:
    """PASO 24. Softmax EXISTENTE (softmax_policy_from_topk), sin cambiar la
    formula -- solo se persiste completa en vez de solo su tamano."""
    scored = [{"move_id": c["move_id"], "score_cp": c["score_cp"]} for c in topk]
    policy = softmax_policy_from_topk(scored, temperature_cp)
    rank_by_move_id = {c["move_id"]: c["rank"] for c in topk}
    return [
        {"row_id": row_id, "move_id": p["move_id"], "probability": p["weight"],
         "rank": rank_by_move_id[p["move_id"]]}
        for p in policy
    ]


def eval_row_from_result(row_id: int, result: dict, depth: int, multipv_configured: int) -> dict:
    """PASO 21-22, 25-26. `result` es el dict devuelto por
    StockfishEnricher.evaluate_human_move (ya incluye best_move + human_eval
    + centipawn_loss + move_quality)."""
    topk = result["topk"]
    best = topk[0] if topk else None

    if best is not None:
        best_type, best_cp, best_mate_in, best_unified = _decompose_eval(
            best["score_cp"], best["is_mate"], best["mate_in"]
        )
    else:
        best_type, best_cp, best_mate_in, best_unified = None, None, None, None

    human_unified = result["human_eval_after_cp"]
    # el "after" ya viene invertido a perspectiva de quien jugo (ver
    # stockfish_enrichment.py::evaluate_human_move) -- para saber si ES un
    # mate literal (y no solo un valor grande) se compara contra el umbral
    # de CP_MATE_SCORE, igual convencion que engine_eval.py (solo lectura).
    human_is_mate = abs(human_unified) > (CP_MATE_SCORE - 1000)
    if human_is_mate:
        human_mate_in_signed = (CP_MATE_SCORE - abs(human_unified))
        human_mate_in_signed = human_mate_in_signed if human_unified > 0 else -human_mate_in_signed
        human_type, human_cp, human_mate_in = "mate", None, int(human_mate_in_signed)
    else:
        human_type, human_cp, human_mate_in = "cp", float(human_unified), None

    return {
        "row_id": row_id, "depth": depth,
        "multipv_configured": multipv_configured, "multipv_returned": len(topk),
        "best_move_id": result["best_move_id"], "best_move_uci": result["best_move_uci"],
        "best_eval_type": best_type, "best_eval_cp": best_cp, "best_eval_mate_in": best_mate_in,
        "best_eval_unified_cp_equivalent": best_unified,
        "human_move_uci": result["human_move_uci"],
        "human_eval_type": human_type, "human_eval_cp": human_cp, "human_eval_mate_in": human_mate_in,
        "human_eval_unified_cp_equivalent": int(human_unified),
        "centipawn_loss": result["centipawn_loss"], "move_quality": result["move_quality"],
    }
