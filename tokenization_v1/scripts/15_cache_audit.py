"""Fase 1 del full enrichment: auditoria de cache ANTES de lanzar Stockfish.
Puramente de lectura sobre shards_v2/<test_name>/ -- 0 llamadas al motor.

Determina si el Zobrist position_id ya existente es SUFICIENTE para cachear
requests al motor, o si hace falta el FEN exacto (que incluye halfmove_clock
/fullmove_number, que python-chess NO incluye en su zobrist estandar). No se
asume nada: se compara nunique(position_id) contra nunique(fen) sobre las
filas elegibles y se reporta la diferencia real encontrada.

Calcula tambien las estadisticas de la clave de cache "hija" (posicion
resultante tras el movimiento humano, usada solo para centipawn_loss), que
es un espacio de requests DISTINTO del de la posicion raiz (multipv=3 vs
multipv=1 implicito en evaluate_human_move).
"""
import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import action_id_to_move  # noqa: E402 (no usado aca, solo referencia de contrato)


def occurrence_distribution(counts: "np.ndarray") -> dict:
    return {
        "singleton": int((counts == 1).sum()),
        "2_times": int((counts == 2).sum()),
        "3_to_5_times": int(((counts >= 3) & (counts <= 5)).sum()),
        "more_than_5_times": int((counts > 5).sum()),
        "max_occurrences": int(counts.max()) if len(counts) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    reports_dir = os.path.abspath(os.path.join(base, config["reports_dir"]))
    os.makedirs(reports_dir, exist_ok=True)

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    cols = ["row_id", "fen", "human_move_uci", "human_move_id", "position_id"]
    tables = [pq.read_table(os.path.join(shard_dir, f), columns=cols) for f in shard_files]
    full = pa.concat_tables(tables).to_pandas()

    n_total = len(full)
    eligible = full[full["human_move_id"] != -1].reset_index(drop=True)
    terminal = full[full["human_move_id"] == -1]
    n_eligible = len(eligible)
    n_terminal = len(terminal)

    # A. eligible_rows
    # B. unique position_id (zobrist) entre eligible_rows
    pid_counts = eligible.groupby("position_id").size().to_numpy()
    n_unique_position_id = len(pid_counts)

    # C. unique FEN exacto entre eligible_rows (= unique engine_request_key ROOT,
    #    dado que la config del motor es constante durante todo el run)
    fen_counts = eligible.groupby("fen").size().to_numpy()
    n_unique_fen = int(eligible["fen"].nunique())

    # Auditoria explicita: es Zobrist suficiente? Si nunique(fen) > nunique(position_id),
    # existen grupos con el MISMO position_id pero FEN distinto (halfmove_clock/fullmove
    # variando) -- Zobrist NO es seguro como clave de cache por si solo.
    zobrist_would_be_sufficient = (n_unique_fen == n_unique_position_id)
    # cuantos grupos position_id contienen mas de un FEN distinto
    fen_per_pid = eligible.groupby("position_id")["fen"].nunique()
    n_pid_groups_with_multiple_fens = int((fen_per_pid > 1).sum())
    max_distinct_fens_in_one_pid_group = int(fen_per_pid.max()) if len(fen_per_pid) else 0

    # D/E/F. reuse potencial ROOT (request_type = root_multipv)
    root_cache_hit_potential = n_eligible - n_unique_fen
    root_reuse_pct = 100.0 * root_cache_hit_potential / n_eligible if n_eligible else 0.0

    # G. distribucion de ocurrencias (sobre FEN exacto, que es la clave real usada)
    root_distribution = occurrence_distribution(fen_counts)

    # ---- clave "hija" (posicion tras el movimiento humano) para CPL/quality ----
    t0 = time.time()
    child_fens = np.empty(n_eligible, dtype=object)
    for i, (fen, uci) in enumerate(zip(eligible["fen"].to_numpy(), eligible["human_move_uci"].to_numpy())):
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(uci))
        child_fens[i] = board.fen()
    elapsed_child = time.time() - t0

    child_series = eligible.assign(child_fen=child_fens)["child_fen"]
    n_unique_child_fen = int(child_series.nunique())
    child_counts = child_series.value_counts().to_numpy()
    child_cache_hit_potential = n_eligible - n_unique_child_fen
    child_reuse_pct = 100.0 * child_cache_hit_potential / n_eligible if n_eligible else 0.0
    child_distribution = occurrence_distribution(child_counts)

    # cuantas filas comparten el MISMO fen raiz pero distinto human_move (no
    # pueden compartir cache de CPL/quality entre si, aunque compartan cache
    # de evaluation/topk/policy/criticality del padre)
    moves_per_root_fen = eligible.groupby("fen")["human_move_uci"].nunique()
    n_root_fen_with_multiple_human_moves = int((moves_per_root_fen > 1).sum())

    report = {
        "test_name": args.test_name,
        "A_eligible_rows": n_eligible,
        "terminal_rows_excluded": n_terminal,
        "total_dataset_rows": n_total,
        "B_unique_position_id_zobrist": n_unique_position_id,
        "C_unique_exact_fen": n_unique_fen,
        "zobrist_alone_sufficient_as_cache_key": zobrist_would_be_sufficient,
        "n_position_id_groups_with_multiple_distinct_fens": n_pid_groups_with_multiple_fens,
        "max_distinct_fens_within_one_position_id_group": max_distinct_fens_in_one_pid_group,
        "audit_conclusion": (
            "Zobrist position_id NO es suficiente por si solo como clave de cache: "
            f"{n_pid_groups_with_multiple_fens} grupos de position_id contienen mas de un "
            "FEN distinto (halfmove_clock/fullmove_number difieren pese a piezas/turno/"
            "enroques/al-paso identicos). engine_request_key usa el FEN EXACTO, no el "
            "position_id, precisamente por esto."
            if not zobrist_would_be_sufficient else
            "En este piloto, zobrist position_id coincide 1:1 con FEN exacto entre las "
            "filas elegibles (no se encontraron grupos con halfmove_clock/fullmove_number "
            "distintos). Aun asi engine_request_key sigue usando el FEN completo, no el "
            "zobrist, por robustez -- no se confia en una coincidencia empirica de esta "
            "corrida particular."
        ),
        "D_root_unique_engine_request_key": n_unique_fen,
        "E_root_cache_hit_potential": root_cache_hit_potential,
        "F_root_reuse_pct": round(root_reuse_pct, 4),
        "G_root_occurrence_distribution": root_distribution,
        "child_human_move_specific": {
            "description": (
                "Clave de cache SEPARADA (request_type=child_single_pv) para la evaluacion "
                "de la posicion resultante tras el movimiento humano (usada solo para "
                "centipawn_loss). NO se reutiliza como si fuera el mismo request que el "
                "analisis multipv de la posicion raiz."
            ),
            "n_unique_child_fen": n_unique_child_fen,
            "cache_hit_potential": child_cache_hit_potential,
            "reuse_pct": round(child_reuse_pct, 4),
            "occurrence_distribution": child_distribution,
            "elapsed_seconds_to_compute": round(elapsed_child, 2),
        },
        "same_root_fen_different_human_move": {
            "n_root_fen_with_multiple_distinct_human_moves": n_root_fen_with_multiple_human_moves,
            "note": (
                "Estas filas SI pueden compartir cache de evaluation/best_move/topk/"
                "soft_policy/engine_criticality (dependen solo de la posicion raiz), "
                "pero NUNCA de centipawn_loss/move_quality (dependen del movimiento "
                "humano especifico)."
            ),
        },
    }

    out_path = os.path.join(reports_dir, f"{args.test_name}_cache_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nEscrito en {out_path}")


if __name__ == "__main__":
    main()
