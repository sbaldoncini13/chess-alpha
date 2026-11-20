"""Pipeline principal de tokenization_enrichment_v3. SOLO LECTURA sobre el
Golden Dataset -- nunca escribe en tokenization_v1/. Dos fases:

  FASE GLOBAL (barata, ~decenas de segundos): sequences, eval_trajectory,
  policy_entropy/margins, wdl, mate_labels, terminal_distance,
  human_consensus/agreement, tablebase. Se calculan una vez sobre TODO el
  dataset (algunas -- sequences, eval_trajectory, terminal_distance --
  necesitan mirar partidas completas; human_consensus necesita agrupar
  GLOBALMENTE por position_id, que puede repetirse entre shards) y despues
  se particionan por row_id para escribirse alineadas a los 9 shards
  Golden.

  FASE SHARDED (cara, ~40-50 min proyectados): position_features.py
  (attack maps, king safety, pawn structure, piece activity, pins/see/
  tactical motifs/threats/space/center) -- por shard, por batch, con
  checkpoint/resume atomico igual criterio que
  tokenization_v1/scripts/16_stockfish_full_enrichment.py (bloques +
  checkpoint inicial ANTES del loop, ya con el fix del bug real
  encontrado ahi).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_trajectory import build_eval_white_pov, compute_eval_trajectory  # noqa: E402
from golden_reader import GoldenDataset  # noqa: E402
from human_consensus import compute_human_consensus, compute_human_engine_agreement  # noqa: E402
from io_utils_v3 import atomic_write_bytes, atomic_write_json, atomic_write_parquet_df  # noqa: E402
from mate_labels import compute_mate_labels  # noqa: E402
from policy_entropy_margins import compute_decision_margins, compute_policy_entropy  # noqa: E402
from position_features import compute_all_position_features  # noqa: E402
from see import compute_see_moves  # noqa: E402
from sequences import compute_sequences  # noqa: E402
from tablebase import compute_tablebase_row, open_tablebase  # noqa: E402
from terminal_distance import compute_terminal_distance  # noqa: E402
from wdl import compute_wdl  # noqa: E402

# tokenizer.py (tokv1.0) importado SOLO LECTURA para move_to_action_id --
# el vocabulario debe seguir siendo EXACTAMENTE el mismo que el Golden, no
# se reimplementa una copia que podria divergir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tokenization_v1", "scripts")))
from tokenizer import move_to_action_id  # noqa: E402

TACTICAL_KEYS_PREFIX_SKIP = {"see_best_see", "see_worst_legal_capture_see"}  # ver nota en split_position_features


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


STRATEGIC_PREFIXES = (
    "mobility_", "white_holes", "black_holes", "white_outpost", "black_outpost",
    "white_occupied_knight", "black_occupied_knight", "white_protected_knight", "black_protected_knight",
    "white_rook_", "black_rook_", "white_connected_rooks", "black_connected_rooks",
    "white_bishop_", "black_bishop_", "white_bad_bishop", "black_bad_bishop",
    "white_knight_", "black_knight_", "white_queen_activity", "black_queen_activity",
    "white_queen_attacked", "black_queen_attacked", "space_", "central_space_", "center_",
)
PAWN_PREFIXES = ("white_isolated", "black_isolated", "white_doubled", "black_doubled", "white_tripled",
                  "black_tripled", "white_backward", "black_backward", "white_connected_pawns",
                  "black_connected_pawns", "white_protected_pawns", "black_protected_pawns",
                  "white_blocked_pawns", "black_blocked_pawns", "white_pawn_islands", "black_pawn_islands",
                  "white_passed", "black_passed", "white_connected_passed", "black_connected_passed",
                  "white_protected_passed", "black_protected_passed", "white_max_passed", "black_max_passed",
                  "white_kingside_pawn", "black_kingside_pawn", "white_queenside_pawn", "black_queenside_pawn",
                  "white_center_pawn", "black_center_pawn")
KING_SAFETY_PREFIXES = ("white_king_", "black_king_", "white_pawn_shield", "black_pawn_shield",
                         "white_open_files", "black_open_files", "white_semi_open_files", "black_semi_open_files",
                         "white_enemy_attackers_near", "black_enemy_attackers_near", "white_queen_proximity",
                         "black_queen_proximity", "white_rook_proximity", "black_rook_proximity",
                         "white_queen_near_king", "black_queen_near_king", "white_rook_near_king",
                         "black_rook_near_king", "white_bishop_queen_diagonal", "black_bishop_queen_diagonal",
                         "king_safety_diff")


def split_position_features(row_id: int, feats: dict):
    """Reparte el dict plano de position_features.py en las tablas
    logicas del manifest V3 (position_tactical / position_strategic /
    pawn_structure / king_safety / see_moves)."""
    pawn = {"row_id": row_id}
    king = {"row_id": row_id}
    strategic = {"row_id": row_id}
    tactical = {"row_id": row_id}

    for k, v in feats.items():
        if k.startswith(PAWN_PREFIXES):
            pawn[k] = v
        elif k.startswith(KING_SAFETY_PREFIXES):
            king[k] = v
        elif k.startswith(STRATEGIC_PREFIXES):
            strategic[k] = v
        else:
            tactical[k] = v

    return tactical, strategic, pawn, king


def process_global_families(gd: GoldenDataset, config: dict, logger: Logger) -> dict:
    positions = gd.positions()
    game_index = gd.game_index()
    engine_eval = gd.engine_eval()
    engine_topk = gd.engine_topk()
    engine_policy = gd.engine_policy()

    t0 = time.time()
    seq = compute_sequences(positions, game_index, max_length=max(config["sequence_lengths"]))
    logger.log(f"sequences: {time.time() - t0:.1f}s")

    t0 = time.time()
    eval_wpov = build_eval_white_pov(positions, engine_eval)
    traj_df = compute_eval_trajectory(positions, game_index, eval_wpov)
    logger.log(f"eval_trajectory: {time.time() - t0:.1f}s")

    t0 = time.time()
    entropy_df = compute_policy_entropy(engine_policy)
    margins_df = compute_decision_margins(engine_topk, config["decision_margin_thresholds_cp"])
    logger.log(f"policy_entropy/margins: {time.time() - t0:.1f}s")

    wdl_out = compute_wdl(engine_eval["best_eval_unified_cp_equivalent"].to_numpy(), engine_eval["best_eval_type"].to_numpy())
    wdl_df = pd.DataFrame({"row_id": engine_eval["row_id"], "win_probability_from_cp": wdl_out["win_probability_from_cp"]})

    mate_df = compute_mate_labels(engine_eval)
    term_df = compute_terminal_distance(positions, game_index)

    t0 = time.time()
    consensus_df, group_stats = compute_human_consensus(positions, config["consensus"]["min_occurrence_for_consensus"])
    agreement_df = compute_human_engine_agreement(
        consensus_df, group_stats, engine_eval, engine_policy,
        config.get("strong_agreement_threshold", 0.7), config["decision_margin_thresholds_cp"]["forced_min_gap"],
    )
    logger.log(f"human_consensus: {time.time() - t0:.1f}s")

    engine_derived = traj_df.merge(entropy_df, on="row_id", how="outer") \
        .merge(margins_df, on="row_id", how="outer") \
        .merge(wdl_df, on="row_id", how="outer") \
        .merge(mate_df, on="row_id", how="outer") \
        .merge(term_df, on="row_id", how="outer")

    human_consensus_full = consensus_df.merge(agreement_df, on="row_id", how="left")

    return {"sequences": seq, "engine_derived": engine_derived, "human_consensus": human_consensus_full}


def write_global_outputs_by_shard(gd: GoldenDataset, global_results: dict, out_dir: str, logger: Logger):
    positions = gd.positions()
    for tag in gd.base_shard_tags():
        shard_row_ids = set(gd.positions_for_shard(tag)["row_id"].tolist())
        shard_dir = os.path.join(out_dir, tag)
        os.makedirs(shard_dir, exist_ok=True)

        ed = global_results["engine_derived"]
        ed_shard = ed[ed["row_id"].isin(shard_row_ids)]
        atomic_write_parquet_df(os.path.join(shard_dir, "engine_derived.parquet"), ed_shard)

        hc = global_results["human_consensus"]
        hc_shard = hc[hc["row_id"].isin(shard_row_ids)]
        atomic_write_parquet_df(os.path.join(shard_dir, "human_consensus.parquet"), hc_shard)

        seq = global_results["sequences"]
        mask = np.isin(seq["row_id"], list(shard_row_ids))
        idx = np.where(mask)[0]
        min_i = idx.min()
        meta = pd.DataFrame({
            "row_id": seq["row_id"][idx], "sequence_available_length": seq["sequence_available_length"][idx],
            "truncated_at_game_start": seq["truncated_at_game_start"][idx],
            "repetition_in_sequence": seq["repetition_in_sequence"][idx],
            "captures_in_sequence": seq["captures_in_sequence"][idx],
            "checks_in_sequence": seq["checks_in_sequence"][idx],
            "shard_local_offset": seq["prev_row_ids_offsets"][idx] - seq["prev_row_ids_offsets"][min_i],
        })
        atomic_write_parquet_df(os.path.join(shard_dir, "sequences_meta.parquet"), meta)
        start = seq["prev_row_ids_offsets"][min_i]
        end = start + int(meta["shard_local_offset"].iloc[-1]) + int(meta["sequence_available_length"].iloc[-1])
        flat_slice = seq["prev_row_ids_flat"][start:end]
        atomic_write_bytes(os.path.join(shard_dir, "sequences_prev_row_ids.bin"),
                            np.asarray(flat_slice, dtype=np.int64).tobytes())

        logger.log(f"{tag}: global outputs escritos ({len(ed_shard)} filas engine_derived).")


def block_paths(blocks_dir, batch_idx):
    tag = f"block_{batch_idx:05d}"
    return {name: os.path.join(blocks_dir, f"{tag}.{name}.parquet")
            for name in ("position_tactical", "position_strategic", "pawn_structure", "king_safety", "see_moves")}


def block_is_valid(paths):
    import pyarrow.parquet as pq
    if not all(os.path.isfile(p) for p in paths.values()):
        return False
    try:
        for p in paths.values():
            pq.read_table(p, columns=[])
        return True
    except Exception:
        return False


def process_shard_position_features(tag: str, df: pd.DataFrame, out_dir: str, config: dict, logger: Logger, tb):
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_out = os.path.join(out_dir, tag)
    blocks_dir = os.path.join(shard_out, "_blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    final_marker = os.path.join(shard_out, "position_tactical.parquet")
    if os.path.isfile(final_marker):
        logger.log(f"{tag}: ya consolidado (position_features), se salta.")
        return

    batch_size = config["batch_size"]
    n_rows = len(df)
    n_batches = (n_rows + batch_size - 1) // batch_size
    t_start = time.time()
    n_processed = 0

    for batch_idx in range(n_batches):
        paths = block_paths(blocks_dir, batch_idx)
        if block_is_valid(paths):
            n_processed += min(batch_size, n_rows - batch_idx * batch_size)
            continue

        lo = batch_idx * batch_size
        hi = min(n_rows, lo + batch_size)
        batch = df.iloc[lo:hi]

        import chess as _chess

        tact_rows, strat_rows, pawn_rows, king_rows, see_rows = [], [], [], [], []
        # itertuples(), no iterrows(): preserva el dtype exacto de cada
        # columna (ver la correccion real en human_consensus.py -- misma
        # clase de bug, mejor no depender de que la mezcla de columnas de
        # este DataFrame en particular fuerce dtype=object).
        for r in batch[["row_id", "fen"]].itertuples(index=False):
            row_id = int(r.row_id)
            board = _chess.Board(r.fen)
            feats = compute_all_position_features(r.fen, config)
            tact, strat, pawn, king = split_position_features(row_id, feats)
            tb_row = compute_tablebase_row(board, tb)
            tact.update({f"tablebase_{k}": v for k, v in tb_row.items()})
            tact_rows.append(tact)
            strat_rows.append(strat)
            pawn_rows.append(pawn)
            king_rows.append(king)
            for mv in compute_see_moves(board, move_to_action_id):
                see_rows.append({"row_id": row_id, **mv})

        atomic_write_parquet_df(paths["position_tactical"], pd.DataFrame(tact_rows))
        atomic_write_parquet_df(paths["position_strategic"], pd.DataFrame(strat_rows))
        atomic_write_parquet_df(paths["pawn_structure"], pd.DataFrame(pawn_rows))
        atomic_write_parquet_df(paths["king_safety"], pd.DataFrame(king_rows))
        atomic_write_parquet_df(paths["see_moves"], pd.DataFrame(see_rows) if see_rows else pd.DataFrame({"row_id": [], "move_id": [], "see_material_units": []}))

        n_processed += len(batch)
        elapsed = time.time() - t_start
        rate = n_processed / elapsed if elapsed > 0 else 0
        eta = (n_rows - n_processed) / rate if rate > 0 else float("inf")
        logger.log(f"{tag} batch {batch_idx + 1}/{n_batches}: procesadas={n_processed}/{n_rows} "
                    f"tasa={rate:.1f} pos/s eta={eta:.0f}s")

    block_files = sorted(f for f in os.listdir(blocks_dir) if f.endswith(".position_tactical.parquet"))
    for name in ("position_tactical", "position_strategic", "pawn_structure", "king_safety", "see_moves"):
        tables = [pq.read_table(os.path.join(blocks_dir, bf.replace("position_tactical", name))) for bf in block_files]
        atomic_write_parquet_df(os.path.join(shard_out, f"{name}.parquet"), pa.concat_tables(tables).to_pandas())
    logger.log(f"{tag}: CONSOLIDADO position_features.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", choices=["global", "sharded", "all"], default="all")
    ap.add_argument("--max-shards", type=int, default=None, help="limitar cantidad de shards procesados (piloto/smoke)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    out_dir = os.path.abspath(os.path.join(config_dir, config["output_dir"]))
    logs_dir = os.path.abspath(os.path.join(config_dir, config["logs_dir"]))
    ckpt_dir = os.path.abspath(os.path.join(config_dir, config["checkpoint_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = Logger(os.path.join(logs_dir, "v3_enrich.log"))
    logger.log(f"=== Inicio v3_enrich stage={args.stage} resume={args.resume} ===")

    gd = GoldenDataset(config, config_dir)

    ckpt_path = os.path.join(ckpt_dir, "checkpoint.json")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    if os.path.isfile(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        if not args.resume:
            logger.log("ERROR: checkpoint existente pero no se paso --resume.")
            sys.exit(1)
        if ckpt.get("config_hash") != config_hash:
            logger.log("ERROR: config_hash no coincide, no se reutiliza silenciosamente.")
            sys.exit(1)
    else:
        if args.resume:
            logger.log("ERROR: --resume pedido pero no existe checkpoint.")
            sys.exit(1)
        ckpt = {"config_hash": config_hash, "global_done": False, "shards_done": []}
        atomic_write_json(ckpt_path, ckpt)

    if args.stage in ("global", "all") and not ckpt.get("global_done"):
        results = process_global_families(gd, config, logger)
        write_global_outputs_by_shard(gd, results, out_dir, logger)
        ckpt["global_done"] = True
        atomic_write_json(ckpt_path, ckpt)
        logger.log("Fase global COMPLETADA.")
    elif args.stage in ("global", "all"):
        logger.log("Fase global ya completada, se salta.")

    if args.stage in ("sharded", "all"):
        tb = open_tablebase(config)
        tags = gd.base_shard_tags()
        if args.max_shards is not None:
            tags = tags[:args.max_shards]
        for tag in tags:
            if tag in ckpt.get("shards_done", []):
                logger.log(f"{tag}: ya en checkpoint, se salta.")
                continue
            df = gd.positions_for_shard(tag)
            process_shard_position_features(tag, df, out_dir, config, logger, tb)
            ckpt.setdefault("shards_done", []).append(tag)
            atomic_write_json(ckpt_path, ckpt)

    logger.log("=== Fin v3_enrich ===")
    logger.close()


if __name__ == "__main__":
    main()
