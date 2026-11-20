"""Reporte final de la prueba piloto: metricas reales + proyecciones a 100k/1M.

Solo lee lo que ya esta consolidado en disco (checkpoint, outputs, rejections).
No vuelve a correr nada ni toca ningun dato de search_tree_v1.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SHARDS_DIR = os.path.join(BASE_DIR, "shards")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MARGIN = 1.20  # 20% de margen conservador para las proyecciones


def percentile(sorted_arr, p):
    if len(sorted_arr) == 0:
        return None
    idx = min(len(sorted_arr) - 1, int(round(p / 100 * (len(sorted_arr) - 1))))
    return float(sorted_arr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--resumption-test-notes", default="")
    args = ap.parse_args()
    test_name = args.test_name

    ckpt_path = os.path.join(CHECKPOINTS_DIR, test_name, "checkpoint.json")
    out_dir = os.path.join(OUTPUTS_DIR, test_name)
    reject_path = os.path.join(SHARDS_DIR, test_name, "rejections.parquet")

    with open(ckpt_path, encoding="utf-8") as f:
        ckpt = json.load(f)

    meta = pq.read_table(os.path.join(out_dir, "metadata.parquet")).to_pandas()
    lengths = np.load(os.path.join(out_dir, "game_lengths.npy"))
    offsets = np.load(os.path.join(out_dir, "game_offsets.npy"))
    results = np.load(os.path.join(out_dir, "results.npy"))
    moves_path = os.path.join(out_dir, "moves_uint16.bin")
    n_tokens = os.path.getsize(moves_path) // 2

    rejections = pq.read_table(reject_path).to_pandas() if os.path.isfile(reject_path) else None

    # --- Integridad ---
    n_games = len(meta)
    dup_hashes = meta["identity_hash"].duplicated().sum()
    sample_ids = sorted(meta["sample_id"].tolist())
    sample_ids_ok = sample_ids == list(range(n_games))
    expected_offsets = np.zeros(n_games, dtype=np.int64)
    if n_games > 1:
        expected_offsets[1:] = np.cumsum(lengths.astype(np.int64))[:-1]
    offsets_ok = np.array_equal(offsets, expected_offsets)
    tokens_sum_ok = int(lengths.sum()) == n_tokens

    sorted_lengths = np.sort(lengths)

    total_elapsed = ckpt["elapsed_seconds_total"]
    wall_clock_total = None
    if "first_started_utc" in ckpt:
        try:
            t0 = time.mktime(time.strptime(ckpt["first_started_utc"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            t1 = time.mktime(time.strptime(ckpt["last_checkpoint_utc"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            wall_clock_total = t1 - t0
        except Exception:
            wall_clock_total = None

    games_per_sec = n_games / total_elapsed if total_elapsed > 0 else 0
    moves_per_sec = n_tokens / total_elapsed if total_elapsed > 0 else 0

    disk_bytes = (
        os.path.getsize(os.path.join(out_dir, "moves_uint16.bin"))
        + os.path.getsize(os.path.join(out_dir, "game_offsets.npy"))
        + os.path.getsize(os.path.join(out_dir, "game_lengths.npy"))
        + os.path.getsize(os.path.join(out_dir, "results.npy"))
        + os.path.getsize(os.path.join(out_dir, "metadata.parquet"))
    )
    bytes_per_game = disk_bytes / n_games if n_games else 0
    bytes_per_move = disk_bytes / n_tokens if n_tokens else 0

    def project(n_target):
        t_est = (n_target / games_per_sec) if games_per_sec > 0 else None
        space_est = bytes_per_game * n_target
        return {
            "tiempo_estimado_horas": round(t_est / 3600, 2) if t_est else None,
            "tiempo_estimado_horas_con_margen_20pct": round(t_est * MARGIN / 3600, 2) if t_est else None,
            "espacio_estimado_gb": round(space_est / (1024 ** 3), 3),
            "espacio_estimado_gb_con_margen_20pct": round(space_est * MARGIN / (1024 ** 3), 3),
        }

    report = {
        "test_name": test_name,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partidas_solicitadas": ckpt["target_valid_games"],
        "partidas_validas_consolidadas": n_games,
        "candidatos_intentados_total": ckpt["next_draw_rank"],
        "partidas_rechazadas_total": ckpt["n_rejected_total"],
        "desglose_rechazos": ckpt["rejection_reason_counts"],
        "plies_totales": int(lengths.sum()),
        "plies_media": float(lengths.mean()) if n_games else None,
        "plies_mediana": percentile(sorted_lengths, 50),
        "plies_p90": percentile(sorted_lengths, 90),
        "plies_max": int(lengths.max()) if n_games else None,
        "tiempo_total_activo_segundos": total_elapsed,
        "tiempo_total_pared_segundos_incluye_interrupcion": wall_clock_total,
        "n_checkpoints_bloques": ckpt["n_blocks_written"],
        "partidas_por_segundo": round(games_per_sec, 3),
        "movimientos_por_segundo": round(moves_per_sec, 1),
        "ram_max_mb": ckpt.get("ram_max_mb"),
        "ram_media_mb": ckpt.get("ram_avg_mb"),
        "n_workers": ckpt["n_workers"],
        "espacio_total_bytes": disk_bytes,
        "espacio_total_mb": round(disk_bytes / (1024 ** 2), 3),
        "bytes_medios_por_partida": round(bytes_per_game, 1),
        "bytes_medios_por_movimiento": round(bytes_per_move, 2),
        "integridad": {
            "sample_id_contiguo_0_a_n-1": sample_ids_ok,
            "identity_hash_duplicados_en_consolidado": int(dup_hashes),
            "offsets_consistentes_con_lengths": offsets_ok,
            "suma_lengths_igual_a_tokens_escritos": tokens_sum_ok,
            "sin_duplicados": bool(dup_hashes == 0),
            "sin_perdida_de_datos": bool(sample_ids_ok and offsets_ok and tokens_sum_ok),
        },
        "resultado_prueba_interrupcion_reanudacion": args.resumption_test_notes or "no documentado en esta corrida",
        "proyeccion_100000_partidas": project(100_000),
        "proyeccion_1000000_partidas": project(1_000_000),
        "status_checkpoint_final": ckpt["status"],
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"{test_name}_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte escrito en {out_path}")


if __name__ == "__main__":
    main()
