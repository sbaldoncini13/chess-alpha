"""#1 y #2 -- PRIORIDAD ALTA. Mapas de ataque compactos + densidad de
atacantes/defensores por casilla. Una sola pasada de `board.attackers()`
sobre las 64 casillas alimenta ambas familias (evita recalcular dos veces).

squares_attacked_white/black: bitboard de 64 bits (python int), 1 si la
casilla esta atacada por AL MENOS una pieza de ese color (semantica
identica a `board.is_attacked_by()`/planos 17-18 de
src/chess_alpha/encoding.py, solo lectura, no importado -- se reimplementa
para mantener tokenization_enrichment_v3/ independiente).

attackers_count_white/black: array uint8[64], cantidad de atacantes por
casilla y color (no solo boolean).
"""
import chess
import numpy as np


def compute_attack_density(board: chess.Board) -> dict:
    attackers_white = np.zeros(64, dtype=np.uint8)
    attackers_black = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        attackers_white[sq] = len(board.attackers(chess.WHITE, sq))
        attackers_black[sq] = len(board.attackers(chess.BLACK, sq))

    bb_white = 0
    bb_black = 0
    for sq in range(64):
        if attackers_white[sq] > 0:
            bb_white |= (1 << sq)
        if attackers_black[sq] > 0:
            bb_black |= (1 << sq)

    contested = int(((attackers_white > 0) & (attackers_black > 0)).sum())

    return {
        # np.uint64 explicito (no int de Python): si se deja como int de
        # Python, pandas infiere el dtype de la columna por lote segun si
        # ALGUNA fila de ese lote en particular tiene el bit 63 (casilla h8)
        # prendido -- unos lotes salen int64 y otros uint64, y
        # pa.concat_tables revienta con ArrowInvalid al consolidar el shard
        # (bug real, shard_00001 batch 51/51, ver log). Con dtype explicito
        # el dtype de columna es consistente entre TODOS los lotes sin
        # importar el valor de cada fila.
        "squares_attacked_white_bitboard": np.uint64(bb_white),
        "squares_attacked_black_bitboard": np.uint64(bb_black),
        "attacked_square_count_white": int((attackers_white > 0).sum()),
        "attacked_square_count_black": int((attackers_black > 0).sum()),
        "attackers_count_white": attackers_white,
        "attackers_count_black": attackers_black,
        "contested_squares": contested,
    }


def bitboard_to_grid(bitboard: int) -> np.ndarray:
    """Reconstruye el mapa 8x8 (fila=rank 0..7, col=file 0..7) desde el
    bitboard de 64 bits, MISMA convencion square=rank*8+file que
    board_to_tensor_v2 (tokenization_v1/scripts/position_state.py, solo
    lectura, no importado)."""
    grid = np.zeros((8, 8), dtype=np.uint8)
    for sq in range(64):
        if bitboard & (1 << sq):
            row, col = divmod(sq, 8)
            grid[row, col] = 1
    return grid
