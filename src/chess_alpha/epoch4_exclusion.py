"""Conjuntos de exclusion (partidas ya usadas en Epoch1/2/3 o en el
validation set congelado) para el Dataset Epoch 4. Reusa exactamente el
mismo metodo que reports/human_games_elo_audit.* y
scripts/audit_human_lichess_elo_bands.py::section_overlap_check:

- Epoch2 / Epoch3_new / Epoch3_midgame: codigo real de partida de Lichess
  (header Site, ej. 'https://lichess.org/MBkWc60s' -> 'MBkWc60s'), extraido
  del PGN crudo conservado de cada epoch (read_headers(), sin parsear
  movimientos). El validation_set congelado es un SPLIT de estos mismos
  datos (confirmado via su columna row_source: general_new/general_midgame
  vienen de epoch3_new/midgame) - excluir todo el codigo de Site del PGN
  crudo de epoch3 ya cubre entrenamiento Y validation de ese origen.
- Epoch1 (lichess_20k): no conserva el PGN crudo ni el header Site en su
  dataset derivado - unico caso donde hace falta reconstruir el hash de la
  secuencia de jugadas (agrupando por game_id, ordenando por move_index)
  para poder comparar. Cubre tambien el 10-20% de epoch3 que es "replay de
  epoch1+2" (esas filas comparten las mismas partidas fisicas de epoch1)."""

import hashlib
from pathlib import Path

import chess.pgn
import pandas as pd


def lichess_codes_from_pgn(path: Path) -> set:
    if not path.exists():
        return set()
    codes = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            h = chess.pgn.read_headers(f)
            if h is None:
                break
            site = h.get("Site") or ""
            if site.startswith("https://lichess.org/"):
                codes.add(site.rsplit("/", 1)[-1])
    return codes


def reconstruct_move_hashes(path: Path, order_col: str, move_col: str = "played_move_uci",
                             game_id_col: str = "game_id") -> set:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=[game_id_col, order_col, move_col])
    df = df.sort_values([game_id_col, order_col])
    hashes = set()
    for gid, g in df.groupby(game_id_col):
        moves_str = " ".join(g[move_col].tolist())
        hashes.add(hashlib.sha256(moves_str.encode("utf-8")).hexdigest())
    return hashes


def build_exclusion_sets(cfg, root_dir: Path) -> dict:
    """Devuelve dict con 'site_codes' (set, para excluir via header Site sin
    releer movimientos) y 'epoch1_move_hashes' (set, unico caso que requiere
    hash de secuencia de jugadas)."""
    epoch2_codes = lichess_codes_from_pgn(root_dir / cfg["exclusion_sources"]["epoch2_raw_pgn"])
    epoch3_new_codes = lichess_codes_from_pgn(root_dir / cfg["exclusion_sources"]["epoch3_new_raw_pgn"])
    epoch3_midgame_codes = lichess_codes_from_pgn(root_dir / cfg["exclusion_sources"]["epoch3_midgame_raw_pgn"])
    site_codes = epoch2_codes | epoch3_new_codes | epoch3_midgame_codes

    epoch1_hashes = reconstruct_move_hashes(root_dir / cfg["exclusion_sources"]["epoch1_positions"], "move_index")

    return {
        "site_codes": site_codes,
        "site_codes_by_source": {"epoch2": epoch2_codes, "epoch3_new": epoch3_new_codes,
                                   "epoch3_midgame": epoch3_midgame_codes},
        "epoch1_move_hashes": epoch1_hashes,
    }
