"""#26, #27 -- consenso humano por posicion + acuerdo humano vs motor.

canonical_position_consensus_key (PASO 26, decision documentada): se usa
`position_id` (zobrist ya persistido en el Golden -- piezas + turno +
enroques + al-paso). Se descarta deliberadamente `halfmove_clock` como
parte de la clave: no cambia que movimientos son legales ni, en la
inmensa mayoria de los casos, cual es la mejor decision humana -- solo
afecta la elegibilidad de reclamar tablas por 50 movimientos, un caso de
borde raro. Agregar halfmove_clock a la clave fragmentaria los grupos de
consenso sin beneficio real. Distinto del hallazgo de
tokenization_v1/reports/test_10000_cache_validation.json (ese problema era
sobre ESTADO INTERNO NO DETERMINISTA del motor entre llamadas, no sobre
si halfmove_clock afecta la decision -- no es el mismo caso).

Jensen-Shannon divergence (metrica estandar, documentada) para
human_engine_distribution_distance -- ver config.score_fidelity_distance_metric.
"""
import numpy as np
import pandas as pd


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def _js_divergence(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum() if pv.sum() > 0 else pv
    qv = qv / qv.sum() if qv.sum() > 0 else qv
    m = 0.5 * (pv + qv)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def compute_human_consensus(positions: pd.DataFrame, min_occurrence: int) -> pd.DataFrame:
    eligible = positions[positions["human_move_id"] != -1][["row_id", "position_id", "human_move_id"]]

    group_stats = {}
    for pid, grp in eligible.groupby("position_id"):
        counts = grp["human_move_id"].value_counts()
        n = len(grp)
        probs = (counts / n).to_dict()
        top_move = counts.idxmax()
        group_stats[pid] = {
            "occurrence_count": n, "unique_human_moves": len(counts),
            "human_policy_entropy": _entropy(counts.to_numpy() / n),
            "consensus_top_move": int(top_move), "consensus_strength": float(counts.max() / n),
            "human_move_probabilities": probs, "has_multiple_observations": n >= min_occurrence,
        }

    # NUNCA usar DataFrame.iterrows() aca: con columnas de distinto ancho
    # entero (row_id int64, position_id uint64, human_move_id int32),
    # iterrows() unifica cada fila en una sola Series y puede volver
    # position_id un float64 -- un uint64 grande pierde precision al pasar
    # por float64 y el lookup en el dict revienta con KeyError (bug real
    # encontrado en produccion, ver commit). Merge vectorizado en su lugar,
    # preserva dtype y ademas es mucho mas rapido sobre 808k filas.
    group_df = pd.DataFrame([
        {"position_id": pid, "occurrence_count": gs["occurrence_count"],
         "unique_human_moves": gs["unique_human_moves"], "human_policy_entropy": gs["human_policy_entropy"],
         "consensus_top_move": gs["consensus_top_move"], "consensus_strength": gs["consensus_strength"],
         "has_multiple_observations": gs["has_multiple_observations"]}
        for pid, gs in group_stats.items()
    ])
    result = eligible[["row_id", "position_id"]].merge(group_df, on="position_id", how="left")
    return result, group_stats


def compute_human_engine_agreement(consensus_df: pd.DataFrame, group_stats: dict, engine_eval: pd.DataFrame,
                                    engine_policy: pd.DataFrame, strong_agreement_threshold: float,
                                    engine_disagreement_gap_threshold: float) -> pd.DataFrame:
    eval_by_row = engine_eval.set_index("row_id")
    policy_by_row = {rid: dict(zip(g["move_id"], g["probability"])) for rid, g in engine_policy.groupby("row_id")}

    eval_rows_set = set(eval_by_row.index)
    best_move_by_row = eval_by_row["best_move_id"].to_dict()

    rows = []
    # itertuples() preserva el dtype de CADA columna (a diferencia de
    # iterrows(), que unifica la fila entera en una Series y puede volver
    # position_id/row_id float64 -- mismo bug real ya corregido arriba en
    # compute_human_consensus, ver esa nota).
    for r in consensus_df.itertuples(index=False):
        rid = r.row_id
        if rid not in eval_rows_set:
            continue
        best_move_id = best_move_by_row[rid]
        top1_agreement = bool(r.consensus_top_move == best_move_id)
        strong_agreement = top1_agreement and r.consensus_strength >= strong_agreement_threshold

        human_dist = group_stats[r.position_id]["human_move_probabilities"]
        engine_dist = policy_by_row.get(rid, {})
        distance = _js_divergence(human_dist, engine_dist) if engine_dist else None

        engine_disagreement = False
        if not top1_agreement and engine_dist and r.consensus_top_move in engine_dist:
            # gap grande = motor "ve" la jugada humana pero la rechaza con fuerza
            engine_disagreement = r.consensus_strength >= strong_agreement_threshold

        rows.append({
            "row_id": rid, "human_engine_top1_agreement": top1_agreement,
            "strong_human_engine_agreement": strong_agreement,
            "human_engine_distribution_distance_js": distance,
            "human_disagreement": not top1_agreement,
            "engine_disagreement": engine_disagreement,
        })
    return pd.DataFrame(rows)
