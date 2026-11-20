"""Pipeline principal de tokenizacion: candidatos -> validacion paralela -> consolidacion
determinista -> checkpoints atomicos -> deteccion segura -> reanudacion.

Arquitectura (ver justificacion completa en la conversacion aprobada):
- El proceso principal NO hace trabajo pesado de CPU: solo coordina, deduplica,
  escribe a disco y hace checkpoint. Cuenta como "liviano", no como worker.
- Hasta 2 procesos worker (ProcessPoolExecutor, contexto 'spawn' explicito en
  Windows) parsean/validan/tokenizan partidas en paralelo.
- Se preserva el orden determinista con una ventana deslizante FIFO: los
  futures se consumen en el MISMO orden en que se enviaron (que es el orden
  de draw_rank), aunque los workers terminen las tareas en otro orden interno.
  Esto es lo que garantiza sample_id fijo y salida reproducible pese a la
  paralelizacion (requisito 7).
- La deduplicacion, la asignacion de sample_id y la decision de aceptar o
  rechazar se hacen SOLO en el proceso principal, en orden estricto de
  draw_rank -- nunca en los workers.
- Checkpoint = exclusivamente lo que ya esta escrito en disco (requisito 9):
  el puntero de reanudacion (next_draw_rank) solo avanza despues de que un
  bloque fue consolidado con exito.
"""
import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_parser import parse_and_validate_game, VALID  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_DIR, "input")
SHARDS_DIR = os.path.join(BASE_DIR, "shards")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

TOKENIZER_VERSION = "tokv1.0"

_STOP_REQUESTED = False


class IntegrityError(Exception):
    """Violacion de una invariante de integridad (continuidad de sample_id,
    hash duplicado en el consolidado, offsets/tokens inconsistentes). Se
    trata como un error no controlado: se guarda reporte y se detiene, nunca
    se intenta reparar automaticamente."""


class ShardWriteError(Exception):
    """Fallo al escribir o verificar un shard/checkpoint atomico."""


class SafetyLimitStop(Exception):
    """Deteccion segura disparada por un control automatico (RAM agregada,
    espacio en disco). Se maneja igual que una detencion pedida por el
    usuario (drena la ventana, checkpoint, sale limpio), no como un error."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _sigint_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[SIGINT] Deteccion segura solicitada -- terminando la ventana en curso, "
          "no se toman candidatos nuevos.", flush=True)


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


def atomic_write_bytes(final_path: str, data: bytes):
    tmp_path = final_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp_path, "rb") as f:
            check = f.read(min(len(data), 4096))
        if check != data[:len(check)]:
            raise ShardWriteError(f"Verificacion de escritura fallida: {final_path}")
    except OSError as exc:
        raise ShardWriteError(f"Error de OS escribiendo {final_path}: {exc}") from exc
    os.replace(tmp_path, final_path)


def atomic_write_json(final_path: str, obj: dict):
    tmp_path = final_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp_path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        if reloaded != obj:
            raise ShardWriteError(f"Verificacion de escritura JSON fallida: {final_path}")
    except OSError as exc:
        raise ShardWriteError(f"Error de OS escribiendo {final_path}: {exc}") from exc
    os.replace(tmp_path, final_path)


def atomic_write_parquet(final_path: str, table: pa.Table):
    tmp_path = final_path + ".tmp"
    try:
        pq.write_table(table, tmp_path)
        check = pq.read_table(tmp_path, columns=[table.schema.names[0]])
        if check.num_rows != table.num_rows:
            raise ShardWriteError(f"Verificacion de parquet fallida: {final_path}")
    except OSError as exc:
        raise ShardWriteError(f"Error de OS escribiendo {final_path}: {exc}") from exc
    os.replace(tmp_path, final_path)


def load_index_arrays(index_path: str):
    table = pq.read_table(index_path, columns=["game_index", "byte_start", "byte_end"])
    game_index = table.column("game_index").to_numpy()
    assert np.array_equal(game_index, np.arange(len(game_index))), (
        "game_index no es secuencial 0..N-1; no se puede indexar por posicion"
    )
    byte_start = table.column("byte_start").to_numpy()
    byte_end = table.column("byte_end").to_numpy()
    return byte_start, byte_end


def free_disk_gb(path: str) -> float:
    return shutil.disk_usage(os.path.splitdrive(os.path.abspath(path))[0] + os.sep).free / (1024 ** 3)


def rss_mb_recursive(proc: psutil.Process) -> float:
    total = 0
    try:
        total += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    return total / (1024 * 1024)


def read_existing_metadata_hashes(outputs_test_dir: str):
    """Reconstruye {identity_hash: sample_id} a partir del metadata.parquet ya
    consolidado en disco (unica fuente de verdad para deduplicacion en un
    resume), y el conteo/ultimo sample_id."""
    meta_path = os.path.join(outputs_test_dir, "metadata.parquet")
    if not os.path.isfile(meta_path):
        return {}, 0
    table = pq.read_table(meta_path, columns=["sample_id", "identity_hash"])
    sample_ids = table.column("sample_id").to_pylist()
    hashes = table.column("identity_hash").to_pylist()
    hash_map = dict(zip(hashes, sample_ids))
    n_accepted = len(sample_ids)
    return hash_map, n_accepted


def worker_task(args):
    pgn_path, byte_start, byte_end, draw_rank, game_index = args
    return parse_and_validate_game(pgn_path, byte_start, byte_end, draw_rank, game_index)


def consolidate_outputs(test_name: str, logger: Logger):
    """Reconstruye outputs/<test>/ concatenando TODOS los shards en orden de
    bloque (que ya esta en orden de sample_id). A la escala de este piloto
    (<=10000 partidas, unos pocos MB de tokens) recompute-desde-cero en cada
    checkpoint es barato y elimina cualquier riesgo de un append parcial."""
    shard_dir = os.path.join(SHARDS_DIR, test_name)
    out_dir = os.path.join(OUTPUTS_DIR, test_name)
    os.makedirs(out_dir, exist_ok=True)

    block_files = sorted(
        f for f in os.listdir(shard_dir) if f.endswith(".moves.bin")
    ) if os.path.isdir(shard_dir) else []

    all_tokens = []
    all_meta_tables = []
    for bf in block_files:
        block_id = bf[len("block_"):-len(".moves.bin")]
        moves_path = os.path.join(shard_dir, bf)
        meta_path = os.path.join(shard_dir, f"block_{block_id}.meta.parquet")
        tokens = np.fromfile(moves_path, dtype=np.uint16)
        all_tokens.append(tokens)
        all_meta_tables.append(pq.read_table(meta_path))

    if not all_tokens:
        return 0

    moves_concat = np.concatenate(all_tokens)
    meta_concat = pa.concat_tables(all_meta_tables)

    # game_offsets / game_lengths / results derivados de metadata, en orden de sample_id
    meta_sorted_idx = np.argsort(meta_concat.column("sample_id").to_numpy())
    meta_concat = meta_concat.take(pa.array(meta_sorted_idx))

    lengths = meta_concat.column("ply_count").to_numpy().astype(np.uint16)
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths)[:-1]
    results = meta_concat.column("result_code").to_numpy().astype(np.int8)

    # Controles de integridad automaticos (requisito 8): continuidad de
    # sample_id y ausencia de hash duplicado en el consolidado. Una violacion
    # aca indica corrupcion real -- se trata como error no controlado, nunca
    # se intenta reparar en el momento.
    sample_ids_sorted = meta_concat.column("sample_id").to_numpy()
    n = len(sample_ids_sorted)
    if not np.array_equal(sample_ids_sorted, np.arange(n)):
        raise IntegrityError(
            f"sample_id no es contiguo 0..{n - 1} en el consolidado de '{test_name}'"
        )
    hashes = meta_concat.column("identity_hash").to_pylist()
    if len(set(hashes)) != len(hashes):
        raise IntegrityError(
            f"identity_hash duplicado detectado dentro del consolidado de '{test_name}'"
        )
    if int(lengths.sum()) != len(moves_concat):
        raise IntegrityError(
            f"Inconsistencia offsets/tokens: suma de lengths={lengths.sum()} vs "
            f"tokens escritos={len(moves_concat)}"
        )

    atomic_write_bytes(os.path.join(out_dir, "moves_uint16.bin"), moves_concat.tobytes())
    atomic_write_bytes(os.path.join(out_dir, "game_offsets.npy"), _npy_bytes(offsets))
    atomic_write_bytes(os.path.join(out_dir, "game_lengths.npy"), _npy_bytes(lengths))
    atomic_write_bytes(os.path.join(out_dir, "results.npy"), _npy_bytes(results))
    atomic_write_parquet(os.path.join(out_dir, "metadata.parquet"), meta_concat)

    return len(lengths)


def _npy_bytes(arr: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


META_SCHEMA = pa.schema([
    ("sample_id", pa.int64()),
    ("game_index", pa.int64()),
    ("draw_rank", pa.int64()),
    ("byte_start", pa.int64()),
    ("byte_end", pa.int64()),
    ("ply_count", pa.int32()),
    ("result_code", pa.int8()),
    ("white_elo", pa.int32()),
    ("black_elo", pa.int32()),
    ("eco", pa.string()),
    ("event", pa.string()),
    ("date", pa.string()),
    ("initial_fen", pa.string()),
    ("variant", pa.string()),
    ("identity_hash", pa.string()),
])

REJECT_SCHEMA = pa.schema([
    ("draw_rank", pa.int64()),
    ("game_index", pa.int64()),
    ("status", pa.string()),
    ("reason", pa.string()),
    ("error_ply", pa.int32()),
    ("error_move", pa.string()),
    ("error_fen", pa.string()),
    ("duplicate_of_sample_id", pa.int64()),
])


def main():
    global _STOP_REQUESTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    test_name = args.test_name
    if test_name not in config["tests"]:
        print(f"test_name '{test_name}' no esta definido en {args.config}")
        sys.exit(1)
    test_cfg = config["tests"][test_name]
    target = test_cfg["target_valid_games"]
    seed = config["seed"]
    n_workers = config["n_workers"]
    assert n_workers <= 2, "n_workers no puede superar 2 (limite duro para no competir con Stockfish)"
    checkpoint_every = test_cfg.get("checkpoint_every_n_games", config["checkpoint_every_n_games"])
    window_size = config.get("candidate_window_size", 8)
    source_pgn = os.path.abspath(os.path.join(os.path.dirname(args.config), config["source_pgn"]))
    ram_limit_mb = config.get("max_ram_gb_hard_limit", 6) * 1024
    disk_free_min_gb = config.get("min_disk_free_gb", 20)

    shard_dir = os.path.join(SHARDS_DIR, test_name)
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, test_name)
    out_dir = os.path.join(OUTPUTS_DIR, test_name)
    reports_dir = os.path.join(BASE_DIR, "reports")
    reject_path = os.path.join(shard_dir, "rejections.parquet")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.json")
    stop_file = os.path.join(ckpt_dir, "STOP_REQUEST")
    log_path = os.path.join(LOGS_DIR, f"tokenization_{test_name}.log")
    error_report_path = os.path.join(reports_dir, f"{test_name}_error_report.json")

    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    def write_error_report(exc: Exception, context: str):
        """Reporte de error separado del checkpoint (nunca se toca el
        checkpoint valido existente -- requisito 9: no reparar automaticamente,
        solo guardar el error y el ultimo estado bueno conocido)."""
        report = {
            "test_name": test_name,
            "context": context,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_known_state_in_memory": {
                "n_accepted": n_accepted,
                "next_draw_rank": next_draw_rank,
                "n_rejected_total": n_rejected_total,
            },
            "checkpoint_path": ckpt_path,
            "nota": (
                "El checkpoint.json en checkpoint_path (si existe) NO fue modificado "
                "por este error -- sigue reflejando el ultimo estado consolidado valido. "
                "No se intento ninguna reparacion automatica."
            ),
        }
        try:
            with open(error_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except OSError:
            pass  # si ni esto se puede escribir, el traceback ya quedo en el log/stderr
        return report

    logger = Logger(log_path)
    logger.log(f"=== Inicio {test_name} | target={target} | n_workers={n_workers} | seed={seed} ===")

    signal.signal(signal.SIGINT, _sigint_handler)

    checkpoint_exists = os.path.isfile(ckpt_path)
    if args.resume and not checkpoint_exists:
        logger.log(f"ERROR: --resume pedido pero no existe checkpoint en {ckpt_path}. Abortando.")
        sys.exit(1)
    if checkpoint_exists and not args.resume:
        logger.log(
            f"ERROR: ya existe un checkpoint para '{test_name}' en {ckpt_path} pero no se paso "
            f"--resume. Para continuar: --resume. Para reiniciar, borrar manualmente el checkpoint "
            f"y los shards de '{test_name}'. No se inicia desde cero silenciosamente."
        )
        sys.exit(1)

    byte_start_arr, byte_end_arr = load_index_arrays(os.path.join(INPUT_DIR, "game_index.parquet"))
    candidate_order = np.load(os.path.join(INPUT_DIR, "candidate_order.npy"))

    accepted_hashes = {}
    n_accepted = 0
    next_draw_rank = 0
    elapsed_prior = 0.0
    rejection_reason_counts = {}
    n_rejected_total = 0
    first_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ram_max_mb_prior = 0.0
    ram_sum_mb_prior = 0.0
    ram_samples_prior = 0

    if args.resume:
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            required_keys = {"seed", "test_name", "next_draw_rank", "n_accepted_consolidated"}
            missing = required_keys - set(ckpt.keys())
            if missing:
                raise ValueError(f"checkpoint incompleto, faltan claves: {sorted(missing)}")
            if ckpt["seed"] != seed:
                raise ValueError(
                    f"semilla del checkpoint ({ckpt['seed']}) no coincide con la config actual ({seed})"
                )
            if ckpt["test_name"] != test_name:
                raise ValueError(
                    f"checkpoint pertenece a '{ckpt['test_name']}', no a '{test_name}'"
                )
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.log(f"ERROR: checkpoint corrupto o incompatible en {ckpt_path}: {exc}")
            write_error_report(exc, context="carga_de_checkpoint_para_resume")
            logger.log("No se intenta reparar automaticamente. Revisar el checkpoint y el reporte de error a mano.")
            logger.close()
            sys.exit(1)
        accepted_hashes, n_accepted = read_existing_metadata_hashes(out_dir)
        next_draw_rank = ckpt["next_draw_rank"]
        elapsed_prior = ckpt.get("elapsed_seconds_total", 0.0)
        rejection_reason_counts = ckpt.get("rejection_reason_counts", {})
        n_rejected_total = ckpt.get("n_rejected_total", 0)
        first_started_utc = ckpt.get("first_started_utc", first_started_utc)
        ram_max_mb_prior = ckpt.get("ram_max_mb", 0.0)
        ram_sum_mb_prior = ckpt.get("ram_sum_mb_accum", 0.0)
        ram_samples_prior = ckpt.get("ram_samples_accum", 0)
        logger.log("Checkpoint encontrado.")
        logger.log(f"Prueba: {test_name}.")
        logger.log(f"Partidas completadas recuperadas: {n_accepted}.")
        logger.log(f"Partidas pendientes: {max(0, target - n_accepted)}.")
        logger.log(f"Ultimo draw_rank confirmado: {next_draw_rank - 1}.")
        logger.log("Reanudando desde la primera partida pendiente.")

    if n_accepted >= target:
        logger.log(f"'{test_name}' ya tiene {n_accepted} >= target {target}. Nada que hacer.")
        logger.close()
        return

    block_tokens = []
    block_meta_rows = []
    block_reject_rows = []
    block_index = sum(1 for f in os.listdir(shard_dir) if f.endswith(".moves.bin")) if os.path.isdir(shard_dir) else 0

    proc = psutil.Process()
    ram_max_mb = ram_max_mb_prior
    ram_sum_mb = ram_sum_mb_prior
    ram_samples = ram_samples_prior

    t_start = time.time()
    last_log_time = t_start
    last_log_count = n_accepted

    def sample_ram():
        nonlocal ram_max_mb, ram_sum_mb, ram_samples
        v = rss_mb_recursive(proc)
        ram_max_mb = max(ram_max_mb, v)
        ram_sum_mb += v
        ram_samples += 1
        return v

    def flush_block(final=False, stop_reason=None):
        nonlocal block_tokens, block_meta_rows, block_reject_rows, block_index, next_draw_rank
        written_this_call = []
        try:
            if block_meta_rows:
                tokens_arr = np.array(block_tokens, dtype=np.uint16)
                moves_path = os.path.join(shard_dir, f"block_{block_index:04d}.moves.bin")
                meta_path = os.path.join(shard_dir, f"block_{block_index:04d}.meta.parquet")
                atomic_write_bytes(moves_path, tokens_arr.tobytes())
                written_this_call.append(moves_path)
                meta_table = pa.table(
                    {name: [row[name] for row in block_meta_rows] for name in META_SCHEMA.names},
                    schema=META_SCHEMA,
                )
                atomic_write_parquet(meta_path, meta_table)
                written_this_call.append(meta_path)
                block_index += 1

            if block_reject_rows:
                existing = pq.read_table(reject_path) if os.path.isfile(reject_path) else pa.table(
                    {name: [] for name in REJECT_SCHEMA.names}, schema=REJECT_SCHEMA
                )
                new_table = pa.table(
                    {name: [row[name] for row in block_reject_rows] for name in REJECT_SCHEMA.names},
                    schema=REJECT_SCHEMA,
                )
                combined = pa.concat_tables([existing, new_table])
                atomic_write_parquet(reject_path, combined)

            n_consolidated = consolidate_outputs(test_name, logger)

            elapsed_total = elapsed_prior + (time.time() - t_start)
            status = (
                "completed" if n_consolidated >= target else
                ("stopped_safely" if final and stop_reason else "in_progress")
            )
            ckpt = {
                "test_name": test_name,
                "tokenizer_version": TOKENIZER_VERSION,
                "seed": seed,
                "n_workers": n_workers,
                "target_valid_games": target,
                "next_draw_rank": next_draw_rank,
                "last_draw_rank_consumed": next_draw_rank - 1,
                "n_accepted_consolidated": n_consolidated,
                "last_sample_id_consolidated": n_consolidated - 1,
                "n_rejected_total": n_rejected_total,
                "rejection_reason_counts": rejection_reason_counts,
                "n_blocks_written": block_index,
                "elapsed_seconds_total": round(elapsed_total, 2),
                "first_started_utc": first_started_utc,
                "last_checkpoint_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config_snapshot": {
                    "n_workers": n_workers, "seed": seed, "checkpoint_every_n_games": checkpoint_every,
                    "ram_limit_mb": ram_limit_mb, "disk_free_min_gb": disk_free_min_gb,
                },
                "ram_max_mb": round(ram_max_mb, 1),
                "ram_avg_mb": round(ram_sum_mb / ram_samples, 1) if ram_samples else 0.0,
                "ram_sum_mb_accum": ram_sum_mb,
                "ram_samples_accum": ram_samples,
                "stop_reason": stop_reason if final else None,
                "status": status,
            }
            atomic_write_json(ckpt_path, ckpt)
            logger.log(
                f"Checkpoint escrito: n_accepted={n_consolidated} next_draw_rank={next_draw_rank} "
                f"rechazadas={n_rejected_total} ram_max_mb={ram_max_mb:.0f}"
            )
            block_tokens = []
            block_meta_rows = []
            block_reject_rows = []
            return n_consolidated
        except Exception:
            # Rollback de los shards de ESTE intento para que un futuro --resume
            # no los vea como huerfanos (nunca reflejados en un checkpoint valido)
            # y los vuelva a mezclar duplicados en la proxima consolidacion.
            for p in written_this_call:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            raise

    mp_context = None
    import multiprocessing
    mp_context = multiprocessing.get_context("spawn")

    candidate_iter = iter(range(next_draw_rank, len(candidate_order)))
    futures = deque()

    def submit_next():
        try:
            rank = next(candidate_iter)
        except StopIteration:
            return False
        gidx = int(candidate_order[rank])
        fut = executor.submit(
            worker_task,
            (source_pgn, int(byte_start_arr[gidx]), int(byte_end_arr[gidx]), rank, gidx),
        )
        futures.append(fut)
        return True

    def handle_result(result):
        """Unica funcion que decide aceptar/rechazar un resultado y hace el
        bookkeeping. Usada tanto en el loop principal como al drenar la
        ventana en una detencion segura -- antes estaban duplicadas y la
        version de drenado se olvidaba de registrar los rechazos."""
        nonlocal next_draw_rank, n_accepted, n_rejected_total
        expected_rank = result["draw_rank"]
        assert expected_rank == next_draw_rank, (
            f"Orden roto: se esperaba draw_rank={next_draw_rank}, llego {expected_rank}"
        )
        next_draw_rank = expected_rank + 1

        if result["status"] == VALID:
            ih = result["identity_hash"]
            if ih in accepted_hashes:
                n_rejected_total += 1
                rejection_reason_counts["duplicada"] = rejection_reason_counts.get("duplicada", 0) + 1
                block_reject_rows.append({
                    "draw_rank": result["draw_rank"], "game_index": result["game_index"],
                    "status": "duplicada", "reason": f"identico a sample_id={accepted_hashes[ih]}",
                    "error_ply": -1, "error_move": "", "error_fen": "",
                    "duplicate_of_sample_id": accepted_hashes[ih],
                })
            else:
                sample_id = n_accepted
                accepted_hashes[ih] = sample_id
                n_accepted += 1
                block_tokens.extend(result["action_ids"])
                we = str(result["headers"].get("WhiteElo", "-1"))
                be = str(result["headers"].get("BlackElo", "-1"))
                block_meta_rows.append({
                    "sample_id": sample_id,
                    "game_index": result["game_index"],
                    "draw_rank": result["draw_rank"],
                    "byte_start": result["byte_start"],
                    "byte_end": result["byte_end"],
                    "ply_count": result["ply_count"],
                    "result_code": result["result_code"],
                    "white_elo": int(we) if we.lstrip("-").isdigit() else -1,
                    "black_elo": int(be) if be.lstrip("-").isdigit() else -1,
                    "eco": result["headers"].get("ECO", ""),
                    "event": result["headers"].get("Event", "")[:200],
                    "date": result["headers"].get("Date", ""),
                    "initial_fen": result["initial_fen"],
                    "variant": result["variant"],
                    "identity_hash": ih,
                })
        else:
            n_rejected_total += 1
            rejection_reason_counts[result["status"]] = rejection_reason_counts.get(result["status"], 0) + 1
            block_reject_rows.append({
                "draw_rank": result["draw_rank"], "game_index": result["game_index"],
                "status": result["status"], "reason": str(result.get("reason"))[:300],
                "error_ply": result.get("error_ply") if result.get("error_ply") is not None else -1,
                "error_move": result.get("error_move") or "",
                "error_fen": result.get("error_fen") or "",
                "duplicate_of_sample_id": -1,
            })
        sample_ram()

    stop_reason = None

    def check_safety_limits(ram_now_mb: float) -> str:
        """Controles automaticos del pipeline principal (requisito 8): RAM
        agregada y espacio libre en disco. No lanza procesos externos, no
        hace auditorias -- son chequeos O(1) dentro del mismo loop."""
        if ram_now_mb > ram_limit_mb:
            return f"ram_excedida ({ram_now_mb:.0f}MB > {ram_limit_mb:.0f}MB)"
        free_gb = free_disk_gb(BASE_DIR)
        if free_gb < disk_free_min_gb:
            return f"espacio_insuficiente ({free_gb:.1f}GB < {disk_free_min_gb}GB)"
        return None

    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            for _ in range(window_size):
                if not submit_next():
                    break

            while futures:
                if n_accepted >= target:
                    break

                handle_result(futures.popleft().result())
                ram_now = sample_ram()

                if len(block_meta_rows) >= checkpoint_every:
                    flush_block()

                now = time.time()
                if (n_accepted - last_log_count >= config["log_every_n_games"]) or (now - last_log_time >= config["log_every_n_seconds"]):
                    elapsed = now - t_start
                    rate = n_accepted / elapsed if elapsed > 0 else 0
                    eta = (target - n_accepted) / rate if rate > 0 else float("inf")
                    free_gb = free_disk_gb(BASE_DIR)
                    logger.log(
                        f"progreso: aceptadas={n_accepted}/{target} rechazadas={n_rejected_total} "
                        f"draw_rank={next_draw_rank} tasa={rate:.1f} partidas/s eta={eta:.0f}s "
                        f"ram_mb={ram_max_mb:.0f}(max) disco_libre_gb={free_gb:.1f}"
                    )
                    last_log_time = now
                    last_log_count = n_accepted

                safety_reason = check_safety_limits(ram_now)
                user_stop = _STOP_REQUESTED or os.path.exists(stop_file)
                if safety_reason:
                    stop_reason = safety_reason
                elif user_stop:
                    stop_reason = "solicitud_usuario"

                if stop_reason:
                    logger.log(
                        f"Deteccion segura ({stop_reason}): no se envian mas candidatos, "
                        f"se drena la ventana en curso."
                    )
                    break

                if n_accepted < target:
                    submit_next()

            # drenar futures ya enviados ("terminar la partida en curso, no empezar una nueva"):
            # usa la MISMA logica de aceptar/rechazar/registrar que el loop principal.
            while futures and n_accepted < target:
                handle_result(futures.popleft().result())

            n_consolidated = flush_block(final=True, stop_reason=stop_reason)

        if os.path.exists(stop_file):
            os.remove(stop_file)

        status = "completed" if n_accepted >= target else ("stopped_safely" if stop_reason else "in_progress")
        if status == "stopped_safely":
            logger.log(f"CHECKPOINT GUARDADO — SEGURO PARA CERRAR (motivo: {stop_reason})")
            logger.log(f"Para reanudar: python 04_run_tokenization.py --config {args.config} "
                        f"--test-name {test_name} --resume")
        elif status == "completed":
            logger.log(f"'{test_name}' completado: {n_accepted} partidas validas y unicas consolidadas.")

        logger.log(f"=== Fin {test_name}: status={status} accepted={n_accepted} rejected={n_rejected_total} ===")
        logger.close()

    except Exception as exc:
        logger.log(f"ERROR NO CONTROLADO ({type(exc).__name__}): {exc}")
        logger.log(
            "Deteniendo sin intentar reparar automaticamente. El checkpoint existente (si lo hay) "
            "no fue modificado por este error y sigue siendo valido para --resume."
        )
        write_error_report(exc, context="loop_principal_de_tokenizacion")
        logger.log(f"=== Fin {test_name}: status=stopped_error accepted={n_accepted} rejected={n_rejected_total} ===")
        logger.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
