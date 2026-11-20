"""#20, #21 -- entropia de Soft Policy + margenes Top-1/2/3. Usa
EXCLUSIVAMENTE engine_policy_target.parquet y engine_topk.parquet YA
PERSISTIDOS del Golden (ENGINE_DERIVED, no se reejecuta Stockfish).

policy_entropy = -sum(p*log2(p)) sobre las probabilidades de esa fila.
normalized_policy_entropy = policy_entropy / log2(n_candidatos) (0 si
  n_candidatos<=1 -- una unica opcion tiene entropia maxima=0 por
  definicion, no hay incertidumbre que medir).

Margenes: se usan los `eval_unified_cp_equivalent` de engine_topk (mismo
valor que ya distingue cp/mate en engine_eval.parquet) MAS los flags
explicitos `topN_is_mate` para que un consumidor nunca confunda un gap
"gigante porque hay mate" con un gap de centipawns real -- ninguno de los
dos numeros se oculta.
"""
import numpy as np
import pandas as pd


def compute_policy_entropy(engine_policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_id, grp in engine_policy.groupby("row_id"):
        p = grp["probability"].to_numpy()
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0
        n = len(grp)
        normalized = entropy / np.log2(n) if n > 1 else 0.0
        sorted_p = np.sort(grp["probability"].to_numpy())[::-1]
        rows.append({
            "row_id": row_id, "policy_entropy": entropy, "normalized_policy_entropy": normalized,
            "top1_probability": float(sorted_p[0]) if len(sorted_p) > 0 else None,
            "top2_probability": float(sorted_p[1]) if len(sorted_p) > 1 else None,
            "top3_probability": float(sorted_p[2]) if len(sorted_p) > 2 else None,
        })
    return pd.DataFrame(rows)


def compute_decision_margins(engine_topk: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    rows = []
    for row_id, grp in engine_topk.groupby("row_id"):
        g = grp.sort_values("rank")
        vals = g["eval_unified_cp_equivalent"].to_numpy()
        is_mate = (g["eval_type"] == "mate").to_numpy()
        top1_top2_gap = float(vals[0] - vals[1]) if len(vals) > 1 else None
        top1_top3_gap = float(vals[0] - vals[2]) if len(vals) > 2 else None
        top2_top3_gap = float(vals[1] - vals[2]) if len(vals) > 2 else None

        if top1_top2_gap is None:
            category = "forced"  # una unica jugada legal disponible -> decision obligada
        elif is_mate[0] or (len(is_mate) > 1 and is_mate[1]):
            category = "forced"  # mate en juego: el margen deja de ser comparable en cp puro
        elif top1_top2_gap <= thresholds["competitive_max_gap"]:
            category = "competitive"  # varias jugadas casi igual de buenas
        elif top1_top2_gap >= thresholds["forced_min_gap"]:
            category = "forced"  # el top1 es tan superior que la decision es obligada
        else:
            category = "clear"

        rows.append({
            "row_id": row_id, "top1_top2_gap": top1_top2_gap, "top1_top3_gap": top1_top3_gap,
            "top2_top3_gap": top2_top3_gap, "top1_is_mate": bool(is_mate[0]) if len(is_mate) > 0 else None,
            "top2_is_mate": bool(is_mate[1]) if len(is_mate) > 1 else None,
            "decision_margin_category": category,
        })
    return pd.DataFrame(rows)
