"""Fase 2 del full enrichment: valida que el pipeline CON cache produce
resultados EXACTAMENTE equivalentes al pipeline SIN cache (el ya validado
en la prueba de 1000 posiciones), sobre una muestra pequena pero
deliberadamente construida para incluir posiciones repetidas.

A = ejecucion SIN cache: StockfishEnricher.evaluate_human_move tal cual
    (stockfish_enrichment.py, sin modificar), una llamada fresca por fila.
B = ejecucion CON cache: engine_cached_analysis.analyse_human_move_cached
    sobre una EngineCache vacia al empezar (se llena durante esta misma
    corrida, generando hits reales sobre las repeticiones de la muestra).

Si hay CUALQUIER diferencia no atribuible a un bug de la key (campo
faltante en el FEN), este script NO recomienda continuar -- deja la
decision (corregir la key vs desactivar cache) para main() y para el
reporte, nunca reintenta corregir solo.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from difficulty_criticality import criticality_score as base_criticality_score  # noqa: E402 (no usado, referencia)
from engine_cache import EngineCache  # noqa: E402
from engine_cached_analysis import analyse_human_move_cached  # noqa: E402
from engine_criticality import engine_criticality_row  # noqa: E402
from engine_records import policy_rows_from_topk, topk_rows_from_analysis  # noqa: E402
from shard_integrity import sha256_of_file  # noqa: E402
from stockfish_enrichment import StockfishEnricher  # noqa: E402

SEED = 20260812


def select_validation_sample(full_df, n_target=400, seed=SEED):
    """Muestra deliberada: grupos de FEN repetido (con y sin mismo human
    move) + singletons por fase, para ejercitar todos los casos pedidos."""
    rng = np.random.default_rng(seed)
    eligible = full_df[full_df["human_move_id"] != -1].reset_index(drop=True)

    fen_counts = eligible.groupby("fen").size()
    repeated_fens = fen_counts[fen_counts >= 2].index.to_numpy()
    rng.shuffle(repeated_fens)

    rows = []
    n_from_repeats = 0
    target_repeats = n_target // 2
    for fen in repeated_fens:
        if n_from_repeats >= target_repeats:
            break
        group = eligible[eligible["fen"] == fen]
        take = group.iloc[: min(4, len(group))]
        rows.append(take)
        n_from_repeats += len(take)

    singleton_fens = fen_counts[fen_counts == 1].index.to_numpy()
    singleton_pool = eligible[eligible["fen"].isin(singleton_fens)]
    remaining = max(0, n_target - n_from_repeats)
    per_phase = remaining // 3
    phase_rows = []
    for phase in ("opening", "middlegame", "endgame"):
        sub = singleton_pool[singleton_pool["game_phase"] == phase]
        k = min(per_phase, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        phase_rows.append(sub.iloc[idx])

    sample = pd_concat(rows + phase_rows)
    sample = sample.drop_duplicates(subset=["row_id"]).sort_values("row_id").reset_index(drop=True)
    return sample


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def topk_signature(topk):
    return [(c["rank"], c["move_uci"], c["score_cp"], c["is_mate"], c["mate_in"]) for c in topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    stockfish_path = os.environ.get(config["stockfish_path_env_var"], "")
    if not stockfish_path:
        print(json.dumps({"status": "skipped", "reason": "STOCKFISH_PATH no configurado"}))
        return

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    reports_dir = os.path.abspath(os.path.join(base, config["reports_dir"]))
    tmp_cache_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name, "_cache_validation_tmp"))
    os.makedirs(tmp_cache_dir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    cols = ["row_id", "fen", "human_move_uci", "human_move_id", "game_phase"]
    tables = [pq.read_table(os.path.join(shard_dir, f), columns=cols) for f in shard_files]
    full = pa.concat_tables(tables).to_pandas()

    sample = select_validation_sample(full, n_target=400)
    n = len(sample)

    depth = config["stockfish_depth"]
    multipv = config["stockfish_multipv"]
    child_multipv = config["stockfish_full"]["child_eval_multipv"]
    threads = config["stockfish_full"]["threads"]
    hash_mb = config["stockfish_full"]["hash_mb"]
    temperature = config["stockfish_soft_policy_temperature_cp"]
    quality_thresholds = config["quality_thresholds_cp"]
    crit_cfg = config["engine_criticality"]

    engine_binary_sha256 = sha256_of_file(stockfish_path)

    # ---- A: sin cache ----
    results_a = {}
    t0 = time.time()
    with StockfishEnricher(stockfish_path, depth=depth, multipv=multipv) as sf:
        for _, r in sample.iterrows():
            results_a[int(r["row_id"])] = sf.evaluate_human_move(r["fen"], r["human_move_uci"], quality_thresholds)
    elapsed_a = time.time() - t0

    # ---- B: con cache (vacia al empezar) ----
    cache_db = os.path.join(tmp_cache_dir, "validation_cache.sqlite")
    if os.path.exists(cache_db):
        os.remove(cache_db)
    engine_version = None
    results_b = {}
    t0 = time.time()
    with StockfishEnricher(stockfish_path, depth=depth, multipv=multipv) as sf:
        engine_version = sf.engine.id.get("name", "unknown")
        cache_meta = {
            "engine_version": engine_version, "engine_binary_sha256": engine_binary_sha256,
            "depth": depth, "multipv_root": multipv, "multipv_child": child_multipv,
            "threads": threads, "hash_mb": hash_mb,
        }
        with EngineCache(cache_db, cache_meta) as cache:
            for _, r in sample.iterrows():
                results_b[int(r["row_id"])] = analyse_human_move_cached(
                    sf, cache, r["fen"], r["human_move_uci"], engine_version, engine_binary_sha256,
                    depth, multipv, child_multipv, threads, hash_mb, quality_thresholds,
                )
            cache_stats = cache.counts()
    elapsed_b = time.time() - t0

    mismatches = []
    for row_id in results_a:
        a, b = results_a[row_id], results_b[row_id]
        if a["best_move_uci"] != b["best_move_uci"]:
            mismatches.append({"row_id": row_id, "field": "best_move_uci", "a": a["best_move_uci"], "b": b["best_move_uci"]})
        if topk_signature(a["topk"]) != topk_signature(b["topk"]):
            mismatches.append({"row_id": row_id, "field": "topk", "a": topk_signature(a["topk"]), "b": topk_signature(b["topk"])})
        if a["centipawn_loss"] != b["centipawn_loss"]:
            mismatches.append({"row_id": row_id, "field": "centipawn_loss", "a": a["centipawn_loss"], "b": b["centipawn_loss"]})
        if a["move_quality"] != b["move_quality"]:
            mismatches.append({"row_id": row_id, "field": "move_quality", "a": a["move_quality"], "b": b["move_quality"]})

        policy_a = policy_rows_from_topk(row_id, a["topk"], temperature)
        policy_b = policy_rows_from_topk(row_id, b["topk"], temperature)
        for pa_, pb_ in zip(policy_a, policy_b):
            if abs(pa_["probability"] - pb_["probability"]) > 1e-6:
                mismatches.append({"row_id": row_id, "field": "policy_probability", "a": pa_, "b": pb_})

        topk_rows_a = topk_rows_from_analysis(row_id, a, depth)
        topk_rows_b = topk_rows_from_analysis(row_id, b, depth)
        crit_a = engine_criticality_row(row_id, topk_rows_a, policy_a, crit_cfg)
        crit_b = engine_criticality_row(row_id, topk_rows_b, policy_b, crit_cfg)
        if crit_a["engine_criticality_score"] != crit_b["engine_criticality_score"]:
            mismatches.append({
                "row_id": row_id, "field": "engine_criticality_score",
                "a": crit_a["engine_criticality_score"], "b": crit_b["engine_criticality_score"],
            })

    report = {
        "test_name": args.test_name, "n_positions_compared": n,
        "n_repeated_fen_groups_included": int((sample.groupby("fen").size() >= 2).sum()),
        "elapsed_seconds_without_cache": round(elapsed_a, 2),
        "elapsed_seconds_with_cache": round(elapsed_b, 2),
        "cache_stats_during_this_validation": cache_stats,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "status": "PASS" if not mismatches else "FAIL",
        "recommendation": (
            "Cache produce resultados identicos al pipeline sin cache sobre la muestra -- "
            "AUTORIZADO usar cache en el full run."
            if not mismatches else
            "Se encontraron diferencias -- NO USAR CACHE hasta corregir la clave. "
            "No se corrige automaticamente aca."
        ),
    }

    out_path = os.path.join(reports_dir, f"{args.test_name}_cache_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nEscrito en {out_path}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
