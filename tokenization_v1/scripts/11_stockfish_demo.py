"""PASO 21-27 (demo de coste computacional). Corre el enriquecimiento
Stockfish SOLO sobre una muestra pequena (config.stockfish_demo_sample_size)
para medir throughput real y proyectar el costo de correrlo sobre el piloto
completo -- nunca se lanza automaticamente sobre todo el dataset (esa
decision queda para una autorizacion posterior y explicita).

Requiere STOCKFISH_PATH en el entorno. Si no esta configurado, termina
limpiamente sin tocar nada (el pipeline base no depende de este script).
"""
import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stockfish_enrichment import StockfishEnricher, softmax_policy_from_topk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    stockfish_path = os.environ.get(config["stockfish_path_env_var"], "")
    if not stockfish_path:
        print(json.dumps({
            "status": "skipped", "reason": f"{config['stockfish_path_env_var']} no configurado en el entorno",
        }, indent=2))
        return

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    os.makedirs(out_dir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    tables = [
        pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "fen", "human_move_uci", "human_move_id"])
        for f in shard_files
    ]
    full = pa.concat_tables(tables).to_pandas()
    full = full[full["human_move_id"] != -1].sort_values("row_id").reset_index(drop=True)

    sample_size = config["stockfish_demo_sample_size"]
    sample = full.iloc[:sample_size]

    depth = config["stockfish_depth"]
    multipv = config["stockfish_multipv"]
    temperature = config["stockfish_soft_policy_temperature_cp"]
    quality_thresholds = config["quality_thresholds_cp"]

    rows = []
    t0 = time.time()
    with StockfishEnricher(stockfish_path, depth=depth, multipv=multipv) as sf:
        for _, r in sample.iterrows():
            result = sf.evaluate_human_move(r["fen"], r["human_move_uci"], quality_thresholds)
            policy = softmax_policy_from_topk(
                [{"move_id": c["move_id"], "score_cp": c["score_cp"]} for c in result["topk"]], temperature
            )
            rows.append({
                "row_id": int(r["row_id"]),
                "engine_depth": depth, "engine_best_move_uci": result["best_move_uci"],
                "engine_best_eval_cp": result["best_eval_cp"], "centipawn_loss": result["centipawn_loss"],
                "move_quality": result["move_quality"], "topk_size": len(result["topk"]),
                "policy_target_size": len(policy),
            })
    elapsed = time.time() - t0

    demo_table = pa.table({k: [row[k] for row in rows] for k in rows[0].keys()}) if rows else None
    demo_path = os.path.join(out_dir, "stockfish_demo.parquet")
    if demo_table is not None:
        pq.write_table(demo_table, demo_path)

    n_total_pilot = len(full)
    rate = len(sample) / elapsed if elapsed > 0 else 0
    projection_hours = (n_total_pilot / rate / 3600) if rate > 0 else None

    result = {
        "status": "ok", "test_name": args.test_name, "depth": depth, "multipv": multipv,
        "n_posiciones_demo": len(sample), "elapsed_seconds": round(elapsed, 2),
        "posiciones_por_segundo": round(rate, 3),
        "n_posiciones_totales_con_movimiento_humano_en_piloto": n_total_pilot,
        "proyeccion_horas_piloto_completo": round(projection_hours, 2) if projection_hours else None,
        "output_path": demo_path if rows else None,
        "nota": (
            "Esta es una MEDICION, no una ejecucion masiva. El piloto completo "
            "de Stockfish NO se lanzo automaticamente -- requiere autorizacion "
            "explicita posterior."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
