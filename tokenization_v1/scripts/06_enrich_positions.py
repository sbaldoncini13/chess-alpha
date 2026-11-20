"""Pipeline principal de enriquecimiento de posiciones (tokenization_schema_v2).

Lee EXCLUSIVAMENTE, en modo solo lectura, los outputs ya consolidados de un
test de tokv1.0 (outputs/<test_name>/: moves_uint16.bin, game_offsets.npy,
game_lengths.npy, results.npy, metadata.parquet). Nunca reabre el PGN crudo,
nunca reparsea partidas, nunca escribe dentro de outputs/<test_name>/.

Dos pasadas (ver SCHEMA_V2.md):
  Pasada A: recorre todas las partidas y acumula {position_id: occurrence
            count} GLOBAL sobre todo el dataset (liviana: solo push/pop +
            zobrist hash, sin tensor ni mascara legal).
  Pasada B: recorre de nuevo (reanudable desde next_game_id del checkpoint),
            produce una fila enriquecida por posicion y la escribe en shards
            inmutables (parquet + binario de legal moves) con
            escritura atomica tmp->rename, igual criterio que tokv1.0.

Ningun archivo de search_v1/, sequence_policy_v1/, ni checkpoints del modelo
se lee ni se escribe desde este script.
"""
import argparse
import json
import os
import signal
import sys
import time

import chess
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from difficulty_criticality import (  # noqa: E402
    criticality_score as calc_criticality_score,
    difficulty_bucket,
    difficulty_score as calc_difficulty_score,
    is_tactical_or_critical,
)
from golden_lock import enforce_golden_lock  # noqa: E402
from io_utils_v2 import atomic_write_json, atomic_write_npy, atomic_write_parquet  # noqa: E402
from legal_moves import legal_action_ids  # noqa: E402
from position_dedup import GlobalOccurrenceCounter, position_id as zobrist_position_id  # noqa: E402
from position_state import (  # noqa: E402
    NO_MOVE_TYPE_FLAGS,
    castling_rights,
    en_passant_square,
    game_phase,
    material_counts,
    material_score,
    move_type_flags,
    position_flags,
)
from repetition import InGameRepetitionTracker  # noqa: E402
from sample_weight import default_sample_weight  # noqa: E402
from tokenizer import action_id_to_move  # noqa: E402
from value_target import value_target_stm  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_VERSION = "tokenization_schema_v2"
PIPELINE_VERSION = "tokv2.0"

POSITIONS_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("game_id", pa.int64()), ("ply", pa.int32()),
    ("fen", pa.string()), ("side_to_move", pa.bool_()),
    ("castling_white_kingside", pa.bool_()), ("castling_white_queenside", pa.bool_()),
    ("castling_black_kingside", pa.bool_()), ("castling_black_queenside", pa.bool_()),
    ("en_passant_square", pa.int8()), ("halfmove_clock", pa.int16()), ("fullmove_number", pa.int16()),
    ("position_id", pa.uint64()), ("position_occurrence_count", pa.int32()),
    ("repetition_count_in_game", pa.int8()), ("legal_move_count", pa.int16()),
    ("human_move_id", pa.int32()), ("human_move_uci", pa.string()),
    ("value_target_stm", pa.int8()), ("game_result_white_pov", pa.int8()),
    ("white_pawns", pa.int8()), ("white_knights", pa.int8()), ("white_bishops", pa.int8()),
    ("white_rooks", pa.int8()), ("white_queens", pa.int8()),
    ("black_pawns", pa.int8()), ("black_knights", pa.int8()), ("black_bishops", pa.int8()),
    ("black_rooks", pa.int8()), ("black_queens", pa.int8()),
    ("material_white", pa.int16()), ("material_black", pa.int16()), ("material_diff", pa.int16()),
    ("game_phase", pa.string()),
    ("is_check", pa.bool_()), ("is_checkmate", pa.bool_()), ("is_stalemate", pa.bool_()),
    ("move_is_capture", pa.bool_()), ("move_is_check", pa.bool_()), ("move_is_checkmate", pa.bool_()),
    ("move_is_promotion", pa.bool_()), ("move_is_castling", pa.bool_()), ("move_is_en_passant", pa.bool_()),
    ("move_is_quiet", pa.bool_()),
    ("white_elo", pa.int32()), ("black_elo", pa.int32()), ("eco", pa.string()),
    ("difficulty_score", pa.float32()), ("difficulty_bucket", pa.string()),
    ("tactical_or_critical", pa.bool_()), ("criticality_score", pa.float32()),
    ("sample_weight", pa.float32()),
])

_STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[SIGINT] Deteccion segura solicitada -- se termina la partida en curso, "
          "no se empieza una nueva, se hace checkpoint y se sale.", flush=True)


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


class V1BaseDataset:
    """Envoltorio de solo lectura sobre outputs/<test_name>/ de tokv1.0."""

    def __init__(self, v1_output_dir: str):
        self.dir = v1_output_dir
        self.moves = np.memmap(os.path.join(v1_output_dir, "moves_uint16.bin"), dtype=np.uint16, mode="r")
        self.game_offsets = np.load(os.path.join(v1_output_dir, "game_offsets.npy"))
        self.game_lengths = np.load(os.path.join(v1_output_dir, "game_lengths.npy"))
        self.results = np.load(os.path.join(v1_output_dir, "results.npy"))
        meta = pq.read_table(os.path.join(v1_output_dir, "metadata.parquet")).to_pandas()
        meta = meta.sort_values("sample_id").reset_index(drop=True)
        assert list(meta["sample_id"]) == list(range(len(meta))), (
            "metadata.parquet de v1 no tiene sample_id contiguo 0..N-1 -- no se puede indexar por game_id"
        )
        self.meta = meta
        self.n_games = len(meta)

    def game_action_ids(self, game_id: int) -> np.ndarray:
        start = int(self.game_offsets[game_id])
        length = int(self.game_lengths[game_id])
        return self.moves[start:start + length]


def pass_a_global_occurrence(dataset: V1BaseDataset, logger: Logger) -> GlobalOccurrenceCounter:
    counter = GlobalOccurrenceCounter()
    t0 = time.time()
    for game_id in range(dataset.n_games):
        initial_fen = dataset.meta.at[game_id, "initial_fen"]
        board = chess.Board(initial_fen)
        counter.add(zobrist_position_id(board))
        for action_id in dataset.game_action_ids(game_id):
            move = action_id_to_move(int(action_id))
            board.push(move)
            counter.add(zobrist_position_id(board))
        if (game_id + 1) % 2000 == 0:
            logger.log(f"[pasada A] posiciones-de-partidas procesadas: {game_id + 1}/{dataset.n_games}")
    elapsed = time.time() - t0
    logger.log(f"[pasada A] completa: {dataset.n_games} partidas, {len(counter)} position_id unicos, {elapsed:.1f}s")
    return counter


def build_rows_for_game(game_id: int, dataset: V1BaseDataset, counter: GlobalOccurrenceCounter,
                         next_row_id: int, difficulty_weights: dict, criticality_thresholds: dict):
    """Genera (rows, legal_ids_per_row) para UNA partida: ply_count + 1 filas."""
    meta_row = dataset.meta.iloc[game_id]
    initial_fen = meta_row["initial_fen"]
    result_white_pov = int(dataset.results[game_id])
    white_elo = int(meta_row["white_elo"])
    black_elo = int(meta_row["black_elo"])
    eco = str(meta_row["eco"])

    action_ids_arr = dataset.game_action_ids(game_id)
    ply_count = len(action_ids_arr)

    board = chess.Board(initial_fen)
    tracker = InGameRepetitionTracker()

    rows = []
    legal_ids_per_row = []
    row_id = next_row_id

    for ply in range(ply_count + 1):
        pid = zobrist_position_id(board)
        rep_count = tracker.observe(board)
        pflags = position_flags(board)
        legal_ids = legal_action_ids(board)
        legal_count = len(legal_ids)

        wc = material_counts(board, chess.WHITE)
        bc = material_counts(board, chess.BLACK)
        w_mat = material_score(board, chess.WHITE)
        b_mat = material_score(board, chess.BLACK)
        phase = game_phase(board)
        stm_white = board.turn == chess.WHITE

        has_move = ply < ply_count
        if has_move:
            move = action_id_to_move(int(action_ids_arr[ply]))
            mflags = move_type_flags(board, move)
            human_move_id = int(action_ids_arr[ply])
            human_move_uci = move.uci()
        else:
            move = None
            mflags = NO_MOVE_TYPE_FLAGS
            human_move_id = -1
            human_move_uci = ""

        diff_score = calc_difficulty_score(legal_count, pflags["is_check"], phase, difficulty_weights)
        crit_score = calc_criticality_score(legal_count, pflags["is_check"], criticality_thresholds)

        row = {
            "row_id": row_id, "game_id": game_id, "ply": ply,
            "fen": board.fen(), "side_to_move": stm_white,
            "castling_white_kingside": board.has_kingside_castling_rights(chess.WHITE),
            "castling_white_queenside": board.has_queenside_castling_rights(chess.WHITE),
            "castling_black_kingside": board.has_kingside_castling_rights(chess.BLACK),
            "castling_black_queenside": board.has_queenside_castling_rights(chess.BLACK),
            "en_passant_square": en_passant_square(board),
            "halfmove_clock": board.halfmove_clock, "fullmove_number": board.fullmove_number,
            "position_id": pid, "position_occurrence_count": counter.count_of(pid),
            "repetition_count_in_game": rep_count, "legal_move_count": legal_count,
            "human_move_id": human_move_id, "human_move_uci": human_move_uci,
            "value_target_stm": value_target_stm(result_white_pov, stm_white),
            "game_result_white_pov": result_white_pov,
            "white_pawns": wc["pawns"], "white_knights": wc["knights"], "white_bishops": wc["bishops"],
            "white_rooks": wc["rooks"], "white_queens": wc["queens"],
            "black_pawns": bc["pawns"], "black_knights": bc["knights"], "black_bishops": bc["bishops"],
            "black_rooks": bc["rooks"], "black_queens": bc["queens"],
            "material_white": w_mat, "material_black": b_mat, "material_diff": w_mat - b_mat,
            "game_phase": phase,
            "is_check": pflags["is_check"], "is_checkmate": pflags["is_checkmate"],
            "is_stalemate": pflags["is_stalemate"],
            "move_is_capture": mflags["capture"], "move_is_check": mflags["check"],
            "move_is_checkmate": mflags["checkmate"], "move_is_promotion": mflags["promotion"],
            "move_is_castling": mflags["castling"], "move_is_en_passant": mflags["en_passant"],
            "move_is_quiet": mflags["quiet"],
            "white_elo": white_elo, "black_elo": black_elo, "eco": eco,
            "difficulty_score": diff_score, "difficulty_bucket": difficulty_bucket(diff_score),
            "tactical_or_critical": is_tactical_or_critical(crit_score), "criticality_score": crit_score,
            "sample_weight": default_sample_weight(),
        }
        rows.append(row)
        legal_ids_per_row.append(legal_ids)
        row_id += 1

        if has_move:
            board.push(move)

    return rows, legal_ids_per_row


def flush_shard(shard_dir: str, shard_index: int, rows: list, legal_ids_per_row: list, logger: Logger):
    cols = {name: [] for name in POSITIONS_SCHEMA.names}
    for row in rows:
        for name in POSITIONS_SCHEMA.names:
            cols[name].append(row[name])
    table = pa.table(cols, schema=POSITIONS_SCHEMA)

    offsets = np.zeros(len(legal_ids_per_row), dtype=np.int64)
    running = 0
    flat_ids = []
    for i, ids in enumerate(legal_ids_per_row):
        offsets[i] = running
        flat_ids.extend(ids)
        running += len(ids)
    legal_arr = np.array(flat_ids, dtype=np.uint16)

    tag = f"{shard_index:05d}"
    pq_path = os.path.join(shard_dir, f"shard_{tag}.positions.parquet")
    bin_path = os.path.join(shard_dir, f"shard_{tag}.legalmoves.bin")
    off_path = os.path.join(shard_dir, f"shard_{tag}.legalmoves_offsets.npy")

    atomic_write_parquet(pq_path, table)
    with open(bin_path + ".tmp", "wb") as f:
        f.write(legal_arr.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(bin_path + ".tmp", bin_path)
    atomic_write_npy(off_path, offsets)

    logger.log(f"shard_{tag} escrito: {len(rows)} posiciones, {len(flat_ids)} action_id legales")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True, help="nombre del test de tokv1.0 a enriquecer (p.ej. test_10000)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-golden-overwrite", action="store_true",
                     help="Ignora GOLDEN_DATASET.lock para --test-name si esta protegido. PELIGROSO -- "
                          "invalida un Golden Dataset congelado. No usar salvo mantenimiento deliberado.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    test_name = args.test_name
    v1_output_dir = os.path.abspath(os.path.join(os.path.dirname(args.config), config["base_dataset"]["v1_outputs_dir"], test_name))
    assert os.path.isdir(v1_output_dir), f"No existen outputs de tokv1.0 para '{test_name}' en {v1_output_dir}"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(args.config), config["output_dir"], test_name))
    shard_dir = os.path.abspath(os.path.join(os.path.dirname(args.config), config["shard_dir"], test_name))
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(args.config), config["checkpoint_dir"], test_name))
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(args.config), config["logs_dir"]))
    for d in (out_dir, shard_dir, ckpt_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "checkpoint.json")
    stop_file = os.path.join(ckpt_dir, "STOP_REQUEST")
    log_path = os.path.join(logs_dir, f"enrich_v2_{test_name}.log")
    logger = Logger(log_path)
    logger.log(f"=== Inicio enriquecimiento v2 | test_name={test_name} | v1_source={v1_output_dir} ===")

    golden_root = os.path.abspath(os.path.join(os.path.dirname(args.config), config.get("golden_dir", "../golden")))
    enforce_golden_lock(golden_root, test_name, args.allow_golden_overwrite, logger)

    signal.signal(signal.SIGINT, _sigint_handler)

    checkpoint_exists = os.path.isfile(ckpt_path)
    if args.resume and not checkpoint_exists:
        logger.log(f"ERROR: --resume pedido pero no existe checkpoint en {ckpt_path}.")
        sys.exit(1)
    if checkpoint_exists and not args.resume:
        logger.log(f"ERROR: ya existe checkpoint para '{test_name}'. Usar --resume o borrar manualmente "
                    f"{ckpt_dir} y {shard_dir} para reiniciar. No se reinicia desde cero en silencio.")
        sys.exit(1)

    dataset = V1BaseDataset(v1_output_dir)
    logger.log(f"Dataset base v1 cargado: {dataset.n_games} partidas.")

    counter = pass_a_global_occurrence(dataset, logger)

    next_game_id = 0
    next_row_id = 0
    next_shard_index = 0
    n_positions_consolidated = 0
    elapsed_prior = 0.0
    first_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.resume:
        with open(ckpt_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        assert ckpt["test_name"] == test_name
        next_game_id = ckpt["next_game_id"]
        next_row_id = ckpt["next_row_id"]
        next_shard_index = ckpt["next_shard_index"]
        n_positions_consolidated = ckpt["n_positions_consolidated"]
        elapsed_prior = ckpt.get("elapsed_seconds_total", 0.0)
        first_started_utc = ckpt.get("first_started_utc", first_started_utc)
        logger.log(f"Checkpoint encontrado: reanudando desde game_id={next_game_id}, "
                    f"row_id={next_row_id}, shard={next_shard_index}.")

    if next_game_id >= dataset.n_games:
        logger.log("Ya completado: no queda ninguna partida por procesar.")
        logger.close()
        return

    shard_size = config["shard_size_positions"]
    difficulty_weights = config["difficulty_weights"]
    criticality_thresholds = config["criticality_thresholds"]

    proc = psutil.Process()
    ram_max_mb = 0.0
    t_start = time.time()
    buffer_rows = []
    buffer_legal = []
    stop_reason = None

    for game_id in range(next_game_id, dataset.n_games):
        rows, legal_ids_per_row = build_rows_for_game(
            game_id, dataset, counter, next_row_id, difficulty_weights, criticality_thresholds
        )
        buffer_rows.extend(rows)
        buffer_legal.extend(legal_ids_per_row)
        next_row_id += len(rows)
        ram_max_mb = max(ram_max_mb, proc.memory_info().rss / (1024 * 1024))

        if (game_id + 1) % config["log_every_n_games"] == 0:
            elapsed = time.time() - t_start
            rate = (game_id + 1 - next_game_id) / elapsed if elapsed > 0 else 0
            logger.log(f"progreso: partidas={game_id + 1}/{dataset.n_games} posiciones_buffer={len(buffer_rows)} "
                        f"tasa={rate:.1f} partidas/s ram_mb={ram_max_mb:.0f}")

        user_stop = _STOP_REQUESTED or os.path.exists(stop_file)
        game_boundary_and_full = len(buffer_rows) >= shard_size

        if game_boundary_and_full or user_stop or game_id == dataset.n_games - 1:
            if buffer_rows:
                flush_shard(shard_dir, next_shard_index, buffer_rows, buffer_legal, logger)
                n_positions_consolidated += len(buffer_rows)
                next_shard_index += 1
                buffer_rows = []
                buffer_legal = []

            elapsed_total = elapsed_prior + (time.time() - t_start)
            done = game_id == dataset.n_games - 1 and not user_stop
            ckpt = {
                "test_name": test_name, "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "v1_source_test_name": test_name, "next_game_id": game_id + 1, "next_row_id": next_row_id,
                "next_shard_index": next_shard_index, "n_positions_consolidated": n_positions_consolidated,
                "n_games_consolidated": game_id + 1, "elapsed_seconds_total": round(elapsed_total, 2),
                "first_started_utc": first_started_utc,
                "last_checkpoint_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ram_max_mb": round(ram_max_mb, 1),
                "status": "completed" if done else ("stopped_safely" if user_stop else "in_progress"),
                "stop_reason": "solicitud_usuario" if user_stop else None,
            }
            atomic_write_json(ckpt_path, ckpt)
            logger.log(f"Checkpoint escrito: next_game_id={ckpt['next_game_id']} "
                        f"n_positions_consolidated={n_positions_consolidated} status={ckpt['status']}")

            if user_stop:
                stop_reason = "solicitud_usuario"
                break

    if os.path.exists(stop_file):
        os.remove(stop_file)

    status = "stopped_safely" if stop_reason else "completed"
    logger.log(f"=== Fin enriquecimiento v2: status={status} posiciones={n_positions_consolidated} "
                f"shards={next_shard_index} ===")
    logger.close()


if __name__ == "__main__":
    main()
