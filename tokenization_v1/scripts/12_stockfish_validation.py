"""Prueba CONTROLADA de Stockfish (correccion de auditoria, puntos 21-27):
corre el enriquecimiento completo (eval + top-k + soft policy + engine
criticality) sobre una muestra determinista y pequena (por defecto 1000
posiciones, config.stockfish_validation.sample_size) para VALIDAR que la
persistencia funciona de punta a punta -- no es un enriquecimiento masivo.

Nunca sobrescribe stockfish_demo.parquet (artefacto historico del demo de
60 posiciones). Escribe en un subdirectorio propio e inequivoco. Nunca
modifica shards_v2/<test_name>/ (solo lectura de row_id/fen/human_move_uci/
game_phase/is_check/tactical_or_critical desde ahi).
"""
import argparse
import json
import os
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_criticality import engine_criticality_row  # noqa: E402
from engine_records import eval_row_from_result, policy_rows_from_topk, topk_rows_from_analysis  # noqa: E402
from engine_schemas import (  # noqa: E402
    CRITICALITY_SCHEMA, EVAL_SCHEMA, POLICY_SCHEMA, TOPK_SCHEMA, table_from_rows as _table_from_rows,
)
from io_utils_v2 import atomic_write_json, atomic_write_parquet  # noqa: E402
from sample_selection import select_stratified_sample  # noqa: E402
from stockfish_enrichment import StockfishEnricher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    stockfish_path = os.environ.get(config["stockfish_path_env_var"], "")
    if not stockfish_path:
        print(json.dumps({"status": "skipped", "reason": f"{config['stockfish_path_env_var']} no configurado"}))
        return

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    vcfg = config["stockfish_validation"]
    subdir = os.path.join(out_dir, vcfg["output_subdir"])
    os.makedirs(subdir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    cols = ["row_id", "fen", "human_move_uci", "human_move_id", "game_phase", "is_check", "tactical_or_critical"]
    tables = [pq.read_table(os.path.join(shard_dir, f), columns=cols) for f in shard_files]
    full = pa.concat_tables(tables).to_pandas()

    n_total_base = len(full)
    pool = full[full["human_move_id"] != -1].reset_index(drop=True) if vcfg.get("exclude_terminal_rows", True) else full
    n_pool = len(pool)

    selected_ids = select_stratified_sample(
        pool, n_total=vcfg["sample_size"], seed=vcfg["seed"],
        n_check_quota=vcfg["n_check_quota"], n_tactical_quota=vcfg["n_tactical_quota"],
    )
    sample = pool[pool["row_id"].isin(selected_ids)].sort_values("row_id").reset_index(drop=True)

    depth = config["stockfish_depth"]
    multipv = config["stockfish_multipv"]
    temperature = config["stockfish_soft_policy_temperature_cp"]
    quality_thresholds = config["quality_thresholds_cp"]
    crit_cfg = config["engine_criticality"]

    eval_rows, topk_rows, policy_rows, crit_rows = [], [], [], []
    errors = []
    proc = psutil.Process()
    ram_max_mb = 0.0

    t0 = time.time()
    engine_id = {}
    engine_threads_default = None
    engine_hash_default = None
    with StockfishEnricher(stockfish_path, depth=depth, multipv=multipv) as sf:
        engine_id = dict(sf.engine.id) if sf.engine.id else {}
        try:
            engine_threads_default = sf.engine.options["Threads"].default
        except Exception:
            pass
        try:
            engine_hash_default = sf.engine.options["Hash"].default
        except Exception:
            pass

        for _, r in sample.iterrows():
            row_id = int(r["row_id"])
            try:
                result = sf.evaluate_human_move(r["fen"], r["human_move_uci"], quality_thresholds)
                eval_rows.append(eval_row_from_result(row_id, result, depth, multipv))
                t_rows = topk_rows_from_analysis(row_id, result, depth)
                topk_rows.extend(t_rows)
                p_rows = policy_rows_from_topk(row_id, result["topk"], temperature)
                policy_rows.extend(p_rows)
                crit_rows.append(engine_criticality_row(row_id, t_rows, p_rows, crit_cfg))
            except Exception as exc:
                errors.append({"row_id": row_id, "error": str(exc)})
            ram_max_mb = max(ram_max_mb, proc.memory_info().rss / (1024 * 1024))

    elapsed = time.time() - t0
    n_ok = len(eval_rows)

    atomic_write_parquet(os.path.join(subdir, "engine_eval.parquet"), _table_from_rows(eval_rows, EVAL_SCHEMA))
    atomic_write_parquet(os.path.join(subdir, "engine_topk.parquet"), _table_from_rows(topk_rows, TOPK_SCHEMA))
    atomic_write_parquet(os.path.join(subdir, "engine_policy_target.parquet"), _table_from_rows(policy_rows, POLICY_SCHEMA))
    atomic_write_parquet(os.path.join(subdir, "engine_criticality.parquet"), _table_from_rows(crit_rows, CRITICALITY_SCHEMA))

    disk_bytes = sum(
        os.path.getsize(os.path.join(subdir, f))
        for f in ("engine_eval.parquet", "engine_topk.parquet", "engine_policy_target.parquet", "engine_criticality.parquet")
    )

    rate = n_ok / elapsed if elapsed > 0 else 0
    projection_hours_818611 = (818611 / rate / 3600) if rate > 0 else None

    run_manifest = {
        "engine_schema_extension_version": config["engine_schema_extension_version"],
        "test_name": args.test_name, "status": "completed",
        "engine_path": stockfish_path, "engine_id": engine_id,
        "engine_threads_default": engine_threads_default, "engine_hash_mb_default": engine_hash_default,
        "depth": depth, "multipv": multipv,
        "sample_size_requested": vcfg["sample_size"], "sample_size_selected": len(selected_ids),
        "sample_size_processed_ok": n_ok, "n_errors": len(errors), "errors": errors,
        "seed": vcfg["seed"], "n_check_quota": vcfg["n_check_quota"], "n_tactical_quota": vcfg["n_tactical_quota"],
        "n_positions_base_total": n_total_base, "n_positions_pool_non_terminal": n_pool,
        "elapsed_seconds": round(elapsed, 3), "positions_per_second": round(rate, 4),
        "seconds_per_position_avg": round(elapsed / n_ok, 4) if n_ok else None,
        "ram_max_mb": round(ram_max_mb, 1),
        "output_dir": subdir, "output_bytes_total": disk_bytes,
        "output_bytes_per_position_avg": round(disk_bytes / n_ok, 2) if n_ok else None,
        "proyeccion_horas_818611_posiciones_NO_EJECUTADO": (
            round(projection_hours_818611, 2) if projection_hours_818611 else None
        ),
        "nota": "Proyeccion informativa unicamente. NO se ejecuto sobre las 818611 posiciones.",
    }
    atomic_write_json(os.path.join(subdir, "run_manifest.json"), run_manifest)

    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
