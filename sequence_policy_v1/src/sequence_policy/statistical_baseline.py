"""Baseline estadistico secuencial: B0 (frecuencia global), B1 (por banda de
ply), B2 (prefijo exacto de apertura), B3 (n-gram de sufijo con backoff),
B4 (interpolacion por soporte).

Decision de tamano tomada con datos reales, no por preferencia teorica (ver
reports/NGRAM_RESOURCE_ESTIMATE.json y reports/STATISTICAL_BASELINE.md):
sobre una muestra de 50.000 partidas de train, los contextos de K>=4 son
>95% singleton (aparecen una sola vez) -- proyectar linealmente el conteo
SIN podar da 13-14GB para K=4..16, muy por encima del limite de 6GB. Podar
con minimum_support=2 reduce esos mismos niveles a 1.8-4.5% de su tamano
crudo. Se usa DuckDB para el GROUP BY + HAVING (no se materializa nunca el
diccionario crudo sin podar en memoria de Python)."""
import os

import duckdb
import numpy as np
import pyarrow as pa

from sequence_policy.constants import PLY_BANDS, RAW_VOCAB_SIZE

PAD = 65535  # sentinel fuera de rango action_id (0-20479); cabe en uint16
K_LEVELS = [2, 4, 8, 12, 16]
MIN_SUPPORT_BY_K = {2: 2, 4: 2, 8: 2, 12: 2, 16: 2}
MAX_CONTINUATIONS_PER_CONTEXT = 20
OPENING_PREFIX_MAX_PLY = 16  # B2: acotado por max_k de context_target (c1..c16 ya
                              # codifica el prefijo completo para ply<=16, con PAD
                              # a la izquierda para posiciones mas tempranas)
OPENING_PREFIX_MIN_SUPPORT = 3


def build_context_target_table(con, games_offsets, games_lengths, tokens_path, max_k=16):
    """Construye (via numpy vectorizado por partida, sin loop por posicion)
    una tabla DuckDB context_target(c1..c16, target, ply) para TODO train.
    c1 = token inmediatamente anterior al target, c16 = el mas lejano."""
    mm = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    n_games = len(games_offsets)

    con.execute(f"""
        CREATE OR REPLACE TABLE context_target (
            {', '.join(f'c{i} USMALLINT' for i in range(1, max_k + 1))},
            target USMALLINT,
            ply INTEGER,
            row_id BIGINT
        )
    """)

    # row_id: indice global 0..n-1 asignado EN PYTHON en el mismo orden en que
    # se procesan las partidas (no ROW_NUMBER() OVER() de SQL sin ORDER BY,
    # que no garantiza reproducir el orden de insercion) -- asi el llamador
    # puede reconstruir row_id = suma_de_longitudes_previas + ply sin ambiguedad.
    cols = [f"c{i}" for i in range(1, max_k + 1)] + ["target", "ply", "row_id"]
    pending_blocks = []
    pending_rows = 0
    FLUSH_THRESHOLD_ROWS = 2_000_000
    global_row_counter = 0

    def flush():
        nonlocal pending_blocks, pending_rows
        if not pending_blocks:
            return
        big = np.concatenate(pending_blocks, axis=0)
        arrow_tbl = pa.table({cols[j]: big[:, j] for j in range(len(cols))})
        con.register("_batch", arrow_tbl)
        con.execute("INSERT INTO context_target SELECT * FROM _batch")
        con.unregister("_batch")
        pending_blocks = []
        pending_rows = 0

    for gi in range(n_games):
        start = int(games_offsets[gi])
        ln = int(games_lengths[gi])
        if ln == 0:
            continue
        game = mm[start:start + ln].astype(np.int32)
        padded = np.concatenate([np.full(max_k, PAD, dtype=np.int32), game])
        # sliding_window_view: fila i = padded[i : i+max_k+1] = contexto(max_k) + target
        windows = np.lib.stride_tricks.sliding_window_view(padded, max_k + 1)
        # windows[i, -1] = game[i] = target de la posicion i
        # windows[i, max_k-1] = token justo anterior (c1) ... windows[i, 0] = el mas lejano (c16)
        ctx_rev = windows[:, :max_k][:, ::-1]  # columna 0 = c1 (mas cercano) ... columna max_k-1 = c16
        targets = windows[:, max_k]
        ply_idx = np.arange(ln, dtype=np.int32)
        row_ids = np.arange(global_row_counter, global_row_counter + ln, dtype=np.int64)
        global_row_counter += ln
        rows_block = np.concatenate(
            [ctx_rev, targets[:, None], ply_idx[:, None], row_ids[:, None]], axis=1
        )
        pending_blocks.append(rows_block)
        pending_rows += ln
        if pending_rows >= FLUSH_THRESHOLD_ROWS:
            flush()
    flush()
    del mm
    return con.execute("SELECT COUNT(*) FROM context_target").fetchone()[0]


def build_ngram_level(con, k, min_support, max_continuations):
    """GROUP BY sobre las primeras k columnas (c1..ck) + target, poda por
    minimum_support, top-K continuaciones por contexto. Devuelve un
    DataFrame/arrow con: context (tupla como string), target, count, total_context_count."""
    cols = ", ".join(f"c{i}" for i in range(1, k + 1))
    q = f"""
        WITH counted AS (
            SELECT {cols}, target, COUNT(*) AS cnt
            FROM context_target
            GROUP BY {cols}, target
        ),
        ctx_total AS (
            SELECT {cols}, SUM(cnt) AS total_cnt
            FROM counted
            GROUP BY {cols}
        ),
        joined AS (
            SELECT c.*, t.total_cnt,
                   ROW_NUMBER() OVER (PARTITION BY {cols} ORDER BY c.cnt DESC) AS rn
            FROM counted c
            JOIN ctx_total t USING ({cols})
            WHERE t.total_cnt >= {min_support}
        )
        SELECT {cols}, target, cnt, total_cnt
        FROM joined
        WHERE rn <= {max_continuations}
        ORDER BY {cols}
    """
    return con.execute(q).fetch_arrow_table()


def build_opening_prefix_table(con, max_ply, min_support):
    """B2: prefijo EXACTO desde el inicio de la partida (no sufijo). Usa las
    columnas c1..c_maxply pero interpretadas como prefijo completo -- solo
    tiene sentido para ply <= max_ply (mas alla, el soporte de un prefijo
    exacto colapsa a 1 de todas formas)."""
    cols = ", ".join(f"c{i}" for i in range(1, max_ply + 1))
    q = f"""
        WITH counted AS (
            SELECT ply, {cols}, target, COUNT(*) AS cnt
            FROM context_target
            WHERE ply < {max_ply}
            GROUP BY ply, {cols}, target
        ),
        ctx_total AS (
            SELECT ply, {cols}, SUM(cnt) AS total_cnt
            FROM counted
            GROUP BY ply, {cols}
        )
        SELECT c.ply, {cols}, c.target, c.cnt, t.total_cnt
        FROM counted c JOIN ctx_total t USING (ply, {cols})
        WHERE t.total_cnt >= {min_support}
        ORDER BY ply, {cols}
    """
    return con.execute(q).fetch_arrow_table()


def global_and_ply_band_freq(con):
    """B0 y B1: agregacion trivial, sin poda (el alfabeto es 20480 como
    maximo, cabe siempre)."""
    global_freq = con.execute(
        "SELECT target, COUNT(*) AS cnt FROM context_target GROUP BY target ORDER BY target"
    ).fetch_arrow_table()

    band_case = " ".join(
        f"WHEN ply >= {lo} AND ply < {hi} THEN '{name}'" if hi is not None
        else f"WHEN ply >= {lo} THEN '{name}'"
        for name, lo, hi in PLY_BANDS
    )
    ply_band_freq = con.execute(f"""
        SELECT CASE {band_case} END AS band, target, COUNT(*) AS cnt
        FROM context_target
        GROUP BY band, target
        ORDER BY band, target
    """).fetch_arrow_table()
    return global_freq, ply_band_freq
