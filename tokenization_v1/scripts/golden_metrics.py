"""Metricas baseline de regresion del Golden Dataset. Compartido entre
18_freeze_golden_dataset.py (las escribe la primera vez) y
20_validate_golden_dataset.py --full (las recalcula y compara). Una sola
implementacion -- nunca se duplica la formula de cada metrica."""
import pyarrow as pa
import pyarrow.parquet as pq
import os


def _percentile_stats(series):
    s = series.dropna().sort_values().to_numpy()
    if len(s) == 0:
        return {}

    def p(q):
        idx = min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))
        return float(s[idx])

    return {"min": float(s[0]), "p1": p(1), "p5": p(5), "median": p(50), "p90": p(90),
            "p95": p(95), "p99": p(99), "max": float(s[-1]), "mean": float(s.mean())}


def compute_baseline_metrics(shard_dir: str, full_dir: str) -> dict:
    base_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    base = pa.concat_tables([pq.read_table(os.path.join(shard_dir, f)) for f in base_files]).to_pandas()

    engine_shard_tags = sorted(
        d for d in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, d)) and d.startswith("shard_")
    )
    ev = pa.concat_tables([
        pq.read_table(os.path.join(full_dir, t, "engine_eval.parquet")) for t in engine_shard_tags
    ]).to_pandas()
    tk = pa.concat_tables([
        pq.read_table(os.path.join(full_dir, t, "engine_topk.parquet")) for t in engine_shard_tags
    ]).to_pandas()
    pt = pa.concat_tables([
        pq.read_table(os.path.join(full_dir, t, "engine_policy_target.parquet")) for t in engine_shard_tags
    ]).to_pandas()
    cr = pa.concat_tables([
        pq.read_table(os.path.join(full_dir, t, "engine_criticality.parquet")) for t in engine_shard_tags
    ]).to_pandas()

    n = len(base)
    eligible = base[base["human_move_id"] != -1]
    terminal = base[base["human_move_id"] == -1]
    unique_positions = int(base["position_id"].nunique())

    return {
        "base": {
            "games": int(base["game_id"].nunique()), "plies": int(len(eligible)),
            "terminal_rows": int(len(terminal)), "total_rows": int(n),
            "unique_positions_by_zobrist": unique_positions,
            "duplicate_occurrence_rows": int((base["position_occurrence_count"] > 1).sum()),
            "number_shards": len(base_files),
        },
        "game_phases": base["game_phase"].value_counts().to_dict(),
        "position_flags": {
            "is_check": int(base["is_check"].sum()), "is_checkmate": int(base["is_checkmate"].sum()),
            "is_stalemate": int(base["is_stalemate"].sum()),
        },
        "move_types": {
            "capture": int(base["move_is_capture"].sum()), "promotion": int(base["move_is_promotion"].sum()),
            "castling": int(base["move_is_castling"].sum()), "en_passant": int(base["move_is_en_passant"].sum()),
            "quiet": int(base["move_is_quiet"].sum()),
        },
        "engine": {
            "topk_total_candidates": int(len(tk)),
            "candidates_per_position_mean": round(len(tk) / len(ev), 4) if len(ev) else None,
            "soft_policy_total_rows": int(len(pt)),
        },
        "move_quality_counts": ev["move_quality"].value_counts().to_dict(),
        "centipawn_loss": _percentile_stats(ev["centipawn_loss"]),
        "engine_criticality": {
            **_percentile_stats(cr["engine_criticality_score"]),
            "percentage_critical": round(float(cr["engine_tactical_or_critical"].mean() * 100), 4),
        },
        "sample_weight": {
            "min": float(base["sample_weight"].min()), "max": float(base["sample_weight"].max()),
            "mean": float(base["sample_weight"].mean()), "n_distinct_values": int(base["sample_weight"].nunique()),
            "pct_equal_1_0": round(float((base["sample_weight"] == 1.0).mean() * 100), 4),
        },
    }
