"""Fase 6: enriquecimiento COMPLETO de Stockfish sobre las 808.611 filas
elegibles (human_move_id != -1) del piloto test_10000.

Cache DESACTIVADA (ver config.stockfish_full.cache_disabled_reason y
reports/<test_name>_cache_validation.json): cada posicion se analiza UNA
vez, con el mismo camino ya validado en 12_stockfish_validation.py /
stockfish_enrichment.py (sin modificar). Esto es exactamente el mismo
codigo que ya paso 15/15 validaciones sobre 1000 posiciones -- solo se
escala a las 808.611 filas elegibles, con sharding/checkpoint/resume.

Nunca escribe dentro de shards_v2/<test_name>/. Salida en
outputs_v2/<test_name>/stockfish_full/<shard_tag>/.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback

import chess.engine
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_criticality import engine_criticality_row  # noqa: E402
from engine_records import eval_row_from_result, policy_rows_from_topk, topk_rows_from_analysis  # noqa: E402
from engine_schemas import CRITICALITY_SCHEMA, EVAL_SCHEMA, POLICY_SCHEMA, TOPK_SCHEMA, table_from_rows  # noqa: E402
from golden_lock import enforce_golden_lock  # noqa: E402
from io_utils_v2 import atomic_write_json, atomic_write_parquet  # noqa: E402
from shard_integrity import sha256_of_file  # noqa: E402
from stockfish_enrichment import StockfishEnricher  # noqa: E402

_STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[SIGINT] Deteccion segura solicitada -- se termina el batch en curso y se sale.", flush=True)


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class RestartableEngine:
    """Envuelve StockfishEnricher con reintentos + un restart del proceso
    del motor si se detecta EngineTerminatedError. Una posicion problematica
    nunca debe tirar abajo una corrida de horas."""

    def __init__(self, engine_path, depth, multipv, max_retries, logger):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self.max_retries = max_retries
        self.logger = logger
        self.sf = None
        self._open()

    def _open(self):
        self.sf = StockfishEnricher(self.engine_path, depth=self.depth, multipv=self.multipv)

    def _restart(self):
        try:
            self.sf.close()
        except Exception:
            pass
        self.logger.log("[RestartableEngine] reiniciando proceso de Stockfish tras error.")
        self._open()

    def evaluate(self, fen, human_move_uci, quality_thresholds):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.sf.evaluate_human_move(fen, human_move_uci, quality_thresholds), None
            except chess.engine.EngineTerminatedError as exc:
                last_exc = exc
                self._restart()
            except Exception as exc:
                last_exc = exc
        return None, last_exc

    def close(self):
        try:
            self.sf.close()
        except Exception:
            pass


def load_eligible_by_shard(shard_dir):
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    result = {}
    for f in shard_files:
        tag = f.split(".")[0]
        t = pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "fen", "human_move_uci", "human_move_id"])
        df = t.to_pandas()
        df = df[df["human_move_id"] != -1].sort_values("row_id").reset_index(drop=True)
        result[tag] = df
    return result


def block_paths(blocks_dir, batch_idx):
    tag = f"block_{batch_idx:05d}"
    return {
        "eval": os.path.join(blocks_dir, f"{tag}.engine_eval.parquet"),
        "topk": os.path.join(blocks_dir, f"{tag}.engine_topk.parquet"),
        "policy": os.path.join(blocks_dir, f"{tag}.engine_policy_target.parquet"),
        "crit": os.path.join(blocks_dir, f"{tag}.engine_criticality.parquet"),
    }


def block_is_valid(paths):
    if not all(os.path.isfile(p) for p in paths.values()):
        return False
    try:
        for p in paths.values():
            pq.read_table(p, columns=[])
        return True
    except Exception:
        return False


def process_shard(shard_tag, df, out_dir, config, logger, engine: RestartableEngine):
    shard_out = os.path.join(out_dir, shard_tag)
    blocks_dir = os.path.join(shard_out, "_blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    errors_path = os.path.join(shard_out, "errors.jsonl")

    final_eval_path = os.path.join(shard_out, "engine_eval.parquet")
    if os.path.isfile(final_eval_path):
        logger.log(f"{shard_tag}: ya consolidado, se salta.")
        return {"status": "already_completed", "n_rows": pq.read_table(final_eval_path, columns=["row_id"]).num_rows}

    batch_size = config["stockfish_full"]["batch_size"]
    depth = config["stockfish_depth"]
    multipv = config["stockfish_multipv"]
    temperature = config["stockfish_soft_policy_temperature_cp"]
    quality_thresholds = config["quality_thresholds_cp"]
    crit_cfg = config["engine_criticality"]
    max_retries = config["stockfish_full"]["max_retries_per_position"]

    n_rows = len(df)
    n_batches = (n_rows + batch_size - 1) // batch_size
    proc = psutil.Process()
    ram_max_mb = 0.0
    n_processed = 0
    n_errors = 0
    t_shard_start = time.time()

    for batch_idx in range(n_batches):
        paths = block_paths(blocks_dir, batch_idx)
        if block_is_valid(paths):
            n_processed += min(batch_size, n_rows - batch_idx * batch_size)
            continue

        lo = batch_idx * batch_size
        hi = min(n_rows, lo + batch_size)
        batch = df.iloc[lo:hi]

        eval_rows, topk_rows, policy_rows, crit_rows = [], [], [], []
        for _, r in batch.iterrows():
            row_id = int(r["row_id"])
            result, exc = engine.evaluate(r["fen"], r["human_move_uci"], quality_thresholds)
            if result is None:
                n_errors += 1
                with open(errors_path, "a", encoding="utf-8") as ef:
                    ef.write(json.dumps({
                        "row_id": row_id, "fen": r["fen"], "human_move_uci": r["human_move_uci"],
                        "error_type": type(exc).__name__, "error_message": str(exc),
                        "traceback": traceback.format_exc(), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }) + "\n")
                continue

            eval_rows.append(eval_row_from_result(row_id, result, depth, multipv))
            t_rows = topk_rows_from_analysis(row_id, result, depth)
            topk_rows.extend(t_rows)
            p_rows = policy_rows_from_topk(row_id, result["topk"], temperature)
            policy_rows.extend(p_rows)
            crit_rows.append(engine_criticality_row(row_id, t_rows, p_rows, crit_cfg))

            ram_max_mb = max(ram_max_mb, proc.memory_info().rss / (1024 * 1024))

        if eval_rows:
            atomic_write_parquet(paths["eval"], table_from_rows(eval_rows, EVAL_SCHEMA))
            atomic_write_parquet(paths["topk"], table_from_rows(topk_rows, TOPK_SCHEMA) if topk_rows else pa.table({n: [] for n in TOPK_SCHEMA.names}, schema=TOPK_SCHEMA))
            atomic_write_parquet(paths["policy"], table_from_rows(policy_rows, POLICY_SCHEMA) if policy_rows else pa.table({n: [] for n in POLICY_SCHEMA.names}, schema=POLICY_SCHEMA))
            atomic_write_parquet(paths["crit"], table_from_rows(crit_rows, CRITICALITY_SCHEMA))
        else:
            # batch entero fallido -- igual se escriben archivos vacios validos
            # para que el resume no reintente indefinidamente sin avanzar
            atomic_write_parquet(paths["eval"], pa.table({n: [] for n in EVAL_SCHEMA.names}, schema=EVAL_SCHEMA))
            atomic_write_parquet(paths["topk"], pa.table({n: [] for n in TOPK_SCHEMA.names}, schema=TOPK_SCHEMA))
            atomic_write_parquet(paths["policy"], pa.table({n: [] for n in POLICY_SCHEMA.names}, schema=POLICY_SCHEMA))
            atomic_write_parquet(paths["crit"], pa.table({n: [] for n in CRITICALITY_SCHEMA.names}, schema=CRITICALITY_SCHEMA))

        n_processed += len(batch)
        elapsed = time.time() - t_shard_start
        rate = n_processed / elapsed if elapsed > 0 else 0
        eta = (n_rows - n_processed) / rate if rate > 0 else float("inf")
        logger.log(
            f"{shard_tag} batch {batch_idx + 1}/{n_batches}: procesadas={n_processed}/{n_rows} "
            f"errores={n_errors} tasa={rate:.2f} pos/s eta_shard={eta:.0f}s ram_mb={ram_max_mb:.0f}"
        )

        if _STOP_REQUESTED:
            logger.log(f"{shard_tag}: deteccion segura -- se sale tras este batch, resumible.")
            return {"status": "stopped_safely", "n_processed": n_processed, "n_errors": n_errors}

    # consolidar
    block_files = sorted(f for f in os.listdir(blocks_dir) if f.endswith(".engine_eval.parquet"))
    eval_tables, topk_tables, policy_tables, crit_tables = [], [], [], []
    for bf in block_files:
        tag = bf[: -len(".engine_eval.parquet")]
        eval_tables.append(pq.read_table(os.path.join(blocks_dir, f"{tag}.engine_eval.parquet")))
        topk_tables.append(pq.read_table(os.path.join(blocks_dir, f"{tag}.engine_topk.parquet")))
        policy_tables.append(pq.read_table(os.path.join(blocks_dir, f"{tag}.engine_policy_target.parquet")))
        crit_tables.append(pq.read_table(os.path.join(blocks_dir, f"{tag}.engine_criticality.parquet")))

    atomic_write_parquet(final_eval_path, pa.concat_tables(eval_tables))
    atomic_write_parquet(os.path.join(shard_out, "engine_topk.parquet"), pa.concat_tables(topk_tables))
    atomic_write_parquet(os.path.join(shard_out, "engine_policy_target.parquet"), pa.concat_tables(policy_tables))
    atomic_write_parquet(os.path.join(shard_out, "engine_criticality.parquet"), pa.concat_tables(crit_tables))

    n_final = pa.concat_tables(eval_tables).num_rows
    logger.log(f"{shard_tag}: CONSOLIDADO -- {n_final} filas ({n_errors} errores persistentes de {n_rows} elegibles).")
    return {"status": "completed", "n_rows": n_final, "n_errors": n_errors, "ram_max_mb": ram_max_mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-golden-overwrite", action="store_true",
                     help="Ignora GOLDEN_DATASET.lock para --test-name si esta protegido. PELIGROSO -- "
                          "invalida un Golden Dataset congelado. No usar salvo mantenimiento deliberado.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    stockfish_path = os.environ.get(config["stockfish_path_env_var"], "")
    if not stockfish_path:
        print(json.dumps({"status": "skipped", "reason": "STOCKFISH_PATH no configurado"}))
        sys.exit(1)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name, config["stockfish_full"]["output_subdir"]))
    logs_dir = os.path.abspath(os.path.join(base, config["logs_dir"]))
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f"stockfish_full_{args.test_name}.log")
    logger = Logger(log_path)

    golden_root = os.path.abspath(os.path.join(base, config.get("golden_dir", "../golden")))
    enforce_golden_lock(golden_root, args.test_name, args.allow_golden_overwrite, logger)

    signal.signal(signal.SIGINT, _sigint_handler)

    ckpt_path = os.path.join(out_dir, "checkpoint.json")
    engine_binary_sha256 = sha256_of_file(stockfish_path)

    config_hash_payload = {
        "depth": config["stockfish_depth"], "multipv": config["stockfish_multipv"],
        "quality_thresholds_cp": config["quality_thresholds_cp"],
        "engine_criticality": config["engine_criticality"],
        "soft_policy_temperature_cp": config["stockfish_soft_policy_temperature_cp"],
        "engine_binary_sha256": engine_binary_sha256,
        "engine_schema_extension_version": config["engine_schema_extension_version"],
        "cache_enabled": config["stockfish_full"]["cache_enabled"],
    }
    import hashlib
    config_hash = hashlib.sha256(json.dumps(config_hash_payload, sort_keys=True).encode()).hexdigest()

    if os.path.isfile(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            existing_ckpt = json.load(f)
        if not args.resume:
            logger.log(f"ERROR: ya existe checkpoint en {ckpt_path} pero no se paso --resume.")
            sys.exit(1)
        if existing_ckpt["config_hash"] != config_hash:
            logger.log(
                f"ERROR: config_hash del checkpoint ({existing_ckpt['config_hash']}) no coincide con la "
                f"configuracion actual ({config_hash}). NO se reutiliza silenciosamente -- borrar el "
                f"checkpoint/output o investigar que cambio."
            )
            sys.exit(1)
        run_id = existing_ckpt["run_id"]
        logger.log(f"Resumiendo run_id={run_id} (config_hash verificado OK).")
    else:
        if args.resume:
            logger.log("ERROR: --resume pedido pero no existe checkpoint.")
            sys.exit(1)
        run_id = f"stockfish_full_{args.test_name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        logger.log(f"=== Inicio nuevo run_id={run_id} ===")
        # Checkpoint INICIAL escrito ANTES de procesar el primer shard: si el
        # proceso se interrumpe a mitad del primer shard (como ya paso una
        # vez por un PermissionError transitorio de Windows durante el
        # os.replace atomico), --resume debe encontrar un checkpoint valido
        # contra el cual verificar config_hash, aunque ningun shard haya
        # terminado todavia. El progreso real dentro de un shard interrumpido
        # sigue derivandose de los block files en disco (process_shard/
        # block_is_valid), no de este archivo -- este checkpoint es la
        # condicion de entrada para poder resumir, no la fuente de verdad
        # del progreso fino.
        atomic_write_json(ckpt_path, {
            "run_id": run_id, "config_hash": config_hash, "test_name": args.test_name,
            "engine_schema_extension_version": config["engine_schema_extension_version"],
            "cache_enabled": config["stockfish_full"]["cache_enabled"],
            "total_eligible_rows": None, "shards": {}, "elapsed_seconds_total": 0.0,
            "last_checkpoint_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "in_progress",
        })

    logger.log(f"Cache habilitada: {config['stockfish_full']['cache_enabled']} "
                f"(motivo: {config['stockfish_full']['cache_disabled_reason'][:120]}...)")

    eligible_by_shard = load_eligible_by_shard(shard_dir)
    shard_tags = sorted(eligible_by_shard.keys())
    total_eligible = sum(len(df) for df in eligible_by_shard.values())
    logger.log(f"Filas elegibles totales: {total_eligible} en {len(shard_tags)} shards.")

    t_start = time.time()
    shard_results = {}
    stopped = False

    for shard_tag in shard_tags:
        if _STOP_REQUESTED:
            stopped = True
            break
        logger.log(f"--- {shard_tag}: {len(eligible_by_shard[shard_tag])} filas elegibles ---")
        engine = RestartableEngine(
            stockfish_path, config["stockfish_depth"], config["stockfish_multipv"],
            config["stockfish_full"]["max_retries_per_position"], logger,
        )
        try:
            result = process_shard(shard_tag, eligible_by_shard[shard_tag], out_dir, config, logger, engine)
        finally:
            engine.close()
        shard_results[shard_tag] = result

        ckpt = {
            "run_id": run_id, "config_hash": config_hash, "test_name": args.test_name,
            "engine_schema_extension_version": config["engine_schema_extension_version"],
            "cache_enabled": config["stockfish_full"]["cache_enabled"],
            "total_eligible_rows": total_eligible,
            "shards": {
                tag: shard_results.get(tag, {"status": "pending"}) for tag in shard_tags
            },
            "elapsed_seconds_total": round(time.time() - t_start, 2),
            "last_checkpoint_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "in_progress",
        }
        atomic_write_json(ckpt_path, ckpt)

        if result.get("status") == "stopped_safely":
            stopped = True
            break

    n_completed_rows = sum(r.get("n_rows", 0) for r in shard_results.values() if r.get("status") in ("completed", "already_completed"))
    n_errors_total = sum(r.get("n_errors", 0) for r in shard_results.values())
    all_shards_done = all(
        shard_results.get(tag, {}).get("status") in ("completed", "already_completed") for tag in shard_tags
    )

    final_status = "completed" if (all_shards_done and not stopped) else ("stopped_safely" if stopped else "incomplete")
    logger.log(f"=== Fin: status={final_status} filas_completadas={n_completed_rows}/{total_eligible} "
                f"errores_totales={n_errors_total} ===")
    if final_status == "stopped_safely":
        logger.log(f"Para reanudar: python 16_stockfish_full_enrichment.py --config {args.config} "
                    f"--test-name {args.test_name} --resume")
    logger.close()

    print(json.dumps({
        "status": final_status, "n_completed_rows": n_completed_rows, "total_eligible": total_eligible,
        "n_errors_total": n_errors_total, "run_id": run_id,
    }, indent=2))


if __name__ == "__main__":
    main()
