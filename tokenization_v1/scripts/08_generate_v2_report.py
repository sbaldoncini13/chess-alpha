"""Genera manifest.json (outputs_v2/<test_name>/) y el reporte final del
piloto v2 (reports/<test_name>_v2_report.json). Solo lee lo que ya esta
consolidado en disco -- no vuelve a correr nada."""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def percentile(sorted_arr, p):
    if len(sorted_arr) == 0:
        return None
    idx = min(len(sorted_arr) - 1, int(round(p / 100 * (len(sorted_arr) - 1))))
    return float(sorted_arr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    ckpt_dir = os.path.abspath(os.path.join(base, config["checkpoint_dir"], args.test_name))
    reports_dir = os.path.abspath(os.path.join(base, config["reports_dir"]))
    v1_dir = os.path.abspath(os.path.join(base, config["base_dataset"]["v1_outputs_dir"], args.test_name))
    os.makedirs(reports_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "checkpoint.json"), encoding="utf-8") as f:
        ckpt = json.load(f)

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    tables = [pq.read_table(os.path.join(shard_dir, f)) for f in shard_files]
    full = pa.concat_tables(tables).to_pandas()
    n = len(full)

    v1_meta = pq.read_table(os.path.join(v1_dir, "metadata.parquet")).to_pandas()
    v1_report_path = os.path.join(reports_dir, f"{args.test_name}_report.json")
    v1_report = json.load(open(v1_report_path, encoding="utf-8")) if os.path.isfile(v1_report_path) else {}

    rows_per_game = full.groupby("game_id").size()
    lengths_v2 = (rows_per_game - 1).to_numpy()  # ply_count reconstruido desde v2
    sorted_lengths = np.sort(lengths_v2)

    disk_bytes = 0
    for f in shard_files:
        disk_bytes += os.path.getsize(os.path.join(shard_dir, f))
        tag = f.split(".")[0]
        disk_bytes += os.path.getsize(os.path.join(shard_dir, f"{tag}.legalmoves.bin"))
        disk_bytes += os.path.getsize(os.path.join(shard_dir, f"{tag}.legalmoves_offsets.npy"))

    phase_counts = full["game_phase"].value_counts().to_dict()
    result_counts_white_pov = v1_meta["result_code"].value_counts().to_dict()

    n_castling_moves = int(full["move_is_castling"].sum())
    n_en_passant_moves = int(full["move_is_en_passant"].sum())
    n_promotion_moves = int(full["move_is_promotion"].sum())
    n_capture_moves = int(full["move_is_capture"].sum())
    n_check_moves = int(full["move_is_check"].sum())
    n_checkmate_positions = int(full["is_checkmate"].sum())
    n_stalemate_positions = int(full["is_stalemate"].sum())
    n_check_positions = int(full["is_check"].sum())

    unique_positions = full["position_id"].nunique()
    dup_positions = int((full["position_occurrence_count"] > 1).sum())

    manifest = {
        "schema_version": "tokenization_schema_v2", "pipeline_version": "tokv2.0",
        "generated_from_v1_test_name": args.test_name,
        "v1_partidas_leidas": int(v1_meta.shape[0]),
        "n_positions_total": int(n), "n_positions_unique_by_zobrist": int(unique_positions),
        "n_positions_duplicated_occurrences": dup_positions,
        "n_shards": len(shard_files), "positions_per_shard_target": config["shard_size_positions"],
        "config_used": config,
        "checkpoint_final": ckpt,
        "stockfish_enabled_in_this_run": False,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    dense_demo_path = os.path.join(out_dir, "dense_tensor_demo.npz")
    dense_demo_info = None
    if os.path.isfile(dense_demo_path):
        npz = np.load(dense_demo_path)
        dense_demo_info = {"shape": list(npz["tensors"].shape), "dtype": str(npz["tensors"].dtype),
                            "path": dense_demo_path}

    stockfish_demo_path = os.path.join(out_dir, "stockfish_demo.parquet")
    stockfish_demo_info = None
    if os.path.isfile(stockfish_demo_path):
        sf_t = pq.read_table(stockfish_demo_path).to_pandas()
        stockfish_demo_info = {"n_posiciones": len(sf_t), "path": stockfish_demo_path}

    report = {
        "test_name": args.test_name, "schema_version": "tokenization_schema_v2", "pipeline_version": "tokv2.0",
        "input": {
            "v1_source_dir": v1_dir, "v1_partidas_candidatas": v1_report.get("candidatos_intentados_total"),
        },
        "validacion": {
            "partidas_validas_leidas_de_v1": int(v1_meta.shape[0]),
            "partidas_rechazadas_en_v1": v1_report.get("partidas_rechazadas_total"),
            "desglose_rechazos_v1": v1_report.get("desglose_rechazos"),
        },
        "tokenizacion": {
            "posiciones_totales_v2": int(n),
            "explicacion_diferencia_con_plies_v1": (
                f"v1 reporta {v1_report.get('plies_totales')} plies (=movimientos jugados). "
                f"v2 guarda ply_count+1 filas por partida (incluye la posicion FINAL tras el "
                f"ultimo movimiento, necesaria para detectar is_checkmate/is_stalemate y para "
                f"el value_target de esa posicion terminal) -> "
                f"{v1_report.get('plies_totales')} + {int(v1_meta.shape[0])} partidas = {n} posiciones."
            ),
            "posiciones_unicas_por_zobrist": int(unique_positions),
            "posiciones_con_occurrence_count_mayor_a_1": dup_positions,
            "plies_media": float(lengths_v2.mean()) if n else None,
            "plies_mediana": percentile(sorted_lengths, 50),
            "plies_p90": percentile(sorted_lengths, 90),
            "plies_max": int(lengths_v2.max()) if n else None,
        },
        "movimientos": {
            "vocab_size": 20480, "vocab_version": "tokv1.0",
            "legal_moves_media": float(full["legal_move_count"].mean()),
            "capturas": n_capture_moves, "jaques_dados": n_check_moves,
            "enroques": n_castling_moves, "al_paso": n_en_passant_moves, "promociones": n_promotion_moves,
        },
        "fases": phase_counts,
        "terminales": {
            "posiciones_en_jaque": n_check_positions,
            "posiciones_jaque_mate": n_checkmate_positions,
            "posiciones_ahogado": n_stalemate_positions,
        },
        "resultados_partidas_white_pov": {str(k): int(v) for k, v in result_counts_white_pov.items()},
        "storage": {
            "n_shards": len(shard_files), "bytes_totales": disk_bytes,
            "mb_totales": round(disk_bytes / (1024 ** 2), 2),
            "bytes_por_posicion_aprox": round(disk_bytes / n, 2) if n else None,
            "formato": "parquet (columnas estructuradas + FEN) + binario uint16 (legal moves ragged, patron v1)",
        },
        "performance": {
            "tiempo_total_activo_segundos": ckpt.get("elapsed_seconds_total"),
            "ram_max_mb": ckpt.get("ram_max_mb"),
            "partidas_por_segundo_aprox": round(
                ckpt["n_games_consolidated"] / ckpt["elapsed_seconds_total"], 2
            ) if ckpt.get("elapsed_seconds_total") else None,
        },
        "dense_tensor_demo": dense_demo_info,
        "stockfish": stockfish_demo_info or {"status": "no ejecutado en esta corrida (enable_stockfish=false por defecto)"},
        "manifest_path": manifest_path,
        "status_checkpoint_final": ckpt.get("status"),
    }

    out_path = os.path.join(reports_dir, f"{args.test_name}_v2_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nReporte escrito en {out_path}")
    print(f"Manifest escrito en {manifest_path}")


if __name__ == "__main__":
    main()
