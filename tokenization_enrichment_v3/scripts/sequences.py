"""#28 -- PRIORIDAD MAXIMA. Historial temporal derivado, SOLO HACIA ATRAS,
nunca cruza limites de partida. No modifica el Transformer, no entrena.

Diseno de storage (documentado tambien en docs/FEATURE_DEFINITIONS.md):
en vez de duplicar 4 tablas (una por history_length en {4,8,16,32}), se
guarda UNA sola ventana de hasta 32 `previous_row_ids` por fila (la mas
larga configurada) -- las longitudes mas cortas son simplemente los
ULTIMOS k elementos de esa misma ventana (4 ⊂ 8 ⊂ 16 ⊂ 32), asi que se
pueden reconstruir por slicing sin volver a calcular nada ni duplicar
almacenamiento. `previous_move_ids` NO se guarda aparte: se obtiene por
join de `previous_row_ids` contra `human_move_id` de positions.parquet del
Golden (ya persistido) -- exactamente la referencia compacta que pide el
encargo, no una copia.

Los flags resumen (repetition/captures/checks) se calculan sobre la
ventana COMPLETA disponible (hasta max(sequence_lengths)) para que sean
utiles como filtro rapido; para un sub-analisis por longitud especifica,
recortar `previous_row_ids` y volver a mirar las columnas del Golden.
"""
import numpy as np
import pandas as pd


def compute_sequences(positions: pd.DataFrame, game_index: pd.DataFrame, max_length: int) -> dict:
    """positions: DataFrame completo (o de un shard) con row_id contiguo
    GLOBALMENTE 0..N-1 (invariante ya verificada del Golden). Devuelve dict
    con arrays alineados por posicion (no por row_id arbitrario) listos
    para construir las tablas de salida."""
    row_ids = positions["row_id"].to_numpy()
    game_ids = positions["game_id"].to_numpy()
    position_ids = positions["position_id"].to_numpy()
    move_is_capture = positions["move_is_capture"].to_numpy().astype(np.int64)
    move_is_check = positions["move_is_check"].to_numpy().astype(np.int64)

    row_id_start_by_game = game_index["row_id_start"]
    row_start_per_row = row_id_start_by_game.loc[game_ids].to_numpy()

    available = np.minimum(max_length, row_ids - row_start_per_row).astype(np.int64)
    truncated_at_game_start = available < max_length

    n = len(row_ids)
    prev_row_ids_flat = []
    offsets = np.zeros(n, dtype=np.int64)
    repetition_in_sequence = np.zeros(n, dtype=bool)
    captures_in_sequence = np.zeros(n, dtype=np.int32)
    checks_in_sequence = np.zeros(n, dtype=np.int32)

    running = 0
    for i in range(n):
        avail = available[i]
        offsets[i] = running
        if avail == 0:
            continue
        start_row = row_ids[i] - avail
        window_row_ids = np.arange(start_row, row_ids[i])  # oldest-first
        prev_row_ids_flat.extend(window_row_ids.tolist())
        running += avail

        # las filas del array "positions" estan ordenadas por row_id 0..N-1,
        # asi que window_row_ids son directamente posiciones validas de indexacion
        window_pos_ids = position_ids[window_row_ids]
        repetition_in_sequence[i] = bool(np.isin(position_ids[i], window_pos_ids).any())
        captures_in_sequence[i] = int(move_is_capture[window_row_ids].sum())
        checks_in_sequence[i] = int(move_is_check[window_row_ids].sum())

    prev_row_ids_arr = np.array(prev_row_ids_flat, dtype=np.int64)

    return {
        "row_id": row_ids,
        "sequence_available_length": available,
        "truncated_at_game_start": truncated_at_game_start,
        "repetition_in_sequence": repetition_in_sequence,
        "captures_in_sequence": captures_in_sequence,
        "checks_in_sequence": checks_in_sequence,
        "prev_row_ids_offsets": offsets,
        "prev_row_ids_flat": prev_row_ids_arr,
        "max_length": max_length,
    }


def expand_sequence(prev_row_ids_flat: np.ndarray, offsets: np.ndarray, available: np.ndarray,
                     idx: int, k: int) -> np.ndarray:
    """Reconstruye los ULTIMOS k previous_row_ids de la fila en posicion
    `idx` (oldest-first), recortando de la ventana maxima ya guardada. k
    debe ser <= max_length usado al construir la ventana."""
    avail = int(available[idx])
    start = int(offsets[idx])
    full_window = prev_row_ids_flat[start:start + avail]
    if k >= avail:
        return full_window
    return full_window[-k:]


def previous_move_ids_for_row(prev_row_ids: np.ndarray, positions_indexed_by_row_id: pd.Series) -> np.ndarray:
    """positions_indexed_by_row_id: Series human_move_id indexada por row_id
    (join, no copia -- ver docstring del modulo)."""
    if len(prev_row_ids) == 0:
        return np.array([], dtype=np.int32)
    return positions_indexed_by_row_id.loc[prev_row_ids].to_numpy()
