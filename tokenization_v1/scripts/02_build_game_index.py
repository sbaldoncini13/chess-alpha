"""Indice liviano de offsets binarios + cabeceras basicas.

Un unico pase secuencial al PGN completo, en modo binario, usando
f.readline()/f.tell() explicitos (no 'for line in f', para evitar cualquier
ambiguedad de buffering con tell()). NO usa python-chess, NO parsea movetext,
NO calcula hashes de identidad -- solo separa cada partida por su offset de
byte de inicio/fin y extrae un puñado de campos de cabecera vía texto plano.
Esto es lo que el plan aprobado llama "indice liviano": permite acceso
aleatorio a cualquier partida sin tener que releer el archivo completo.

Escribe input/game_index.parquet incrementalmente (por lotes) para no
acumular las ~4.47M filas en memoria de golpe.
"""
import json
import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_PGN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "AJ-OTB-PGN-000.pgn"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "input"))
OUT_PATH = os.path.join(OUT_DIR, "game_index.parquet")
META_PATH = os.path.join(OUT_DIR, "game_index_manifest.json")

BATCH_SIZE = 250_000
HEADER_LINE_RE = re.compile(rb'^\[(\w+)\s+"(.*)"\]\s*$')

SCHEMA = pa.schema([
    ("game_index", pa.int64()),
    ("byte_start", pa.int64()),
    ("byte_end", pa.int64()),
    ("event", pa.string()),
    ("date", pa.string()),
    ("result", pa.string()),
    ("plycount_header", pa.int32()),
    ("white_elo", pa.int32()),
    ("black_elo", pa.int32()),
    ("eco", pa.string()),
    ("has_setup_fen", pa.bool_()),
    ("variant", pa.string()),
    ("header_ok", pa.bool_()),
])


def parse_header_lines(header_lines: list) -> dict:
    headers = {}
    for raw in header_lines:
        m = HEADER_LINE_RE.match(raw.strip())
        if m:
            key = m.group(1).decode("utf-8", errors="replace")
            val = m.group(2).decode("utf-8", errors="replace")
            headers[key] = val
    return headers


def to_int_or(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_record(game_index: int, byte_start: int, byte_end: int, header_lines: list) -> dict:
    h = parse_header_lines(header_lines)
    return {
        "game_index": game_index,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "event": h.get("Event", "")[:200],
        "date": h.get("Date", ""),
        "result": h.get("Result", ""),
        "plycount_header": to_int_or(h.get("PlyCount"), -1),
        "white_elo": to_int_or(h.get("WhiteElo"), -1),
        "black_elo": to_int_or(h.get("BlackElo"), -1),
        "eco": h.get("ECO", ""),
        "has_setup_fen": h.get("SetUp") == "1" and bool(h.get("FEN")),
        "variant": h.get("Variant", ""),
        "header_ok": "Event" in h and "Result" in h,
    }


def flush_batch(writer, batch_records):
    if not batch_records:
        return
    cols = {name: [] for name in SCHEMA.names}
    for rec in batch_records:
        for name in SCHEMA.names:
            cols[name].append(rec[name])
    table = pa.table(cols, schema=SCHEMA)
    writer.write_table(table)


def main():
    assert os.path.isfile(SOURCE_PGN), f"No existe: {SOURCE_PGN}"
    os.makedirs(OUT_DIR, exist_ok=True)

    tmp_out = OUT_PATH + ".tmp"
    t0 = time.time()
    n_games = 0
    batch = []

    writer = pq.ParquetWriter(tmp_out, SCHEMA)
    try:
        with open(SOURCE_PGN, "rb") as f:
            pos = f.tell()
            line = f.readline()
            current_start = None
            header_lines = []
            while line:
                next_pos = f.tell()
                stripped = line.strip()
                if stripped.startswith(b"[Event "):
                    if current_start is not None:
                        rec = make_record(n_games, current_start, pos, header_lines)
                        batch.append(rec)
                        n_games += 1
                        if len(batch) >= BATCH_SIZE:
                            flush_batch(writer, batch)
                            batch = []
                            elapsed = time.time() - t0
                            print(f"  {n_games} partidas indexadas ({elapsed:.1f}s)")
                    current_start = pos
                    header_lines = [line]
                elif stripped.startswith(b"[") and current_start is not None:
                    header_lines.append(line)
                pos = next_pos
                line = f.readline()

            if current_start is not None:
                rec = make_record(n_games, current_start, pos, header_lines)
                batch.append(rec)
                n_games += 1
            flush_batch(writer, batch)
    finally:
        writer.close()

    # Verificacion: el archivo temporal debe ser legible antes de promoverlo.
    check_table = pq.read_table(tmp_out, columns=["game_index"])
    assert check_table.num_rows == n_games, (
        f"Verificacion fallida: {check_table.num_rows} filas leidas vs {n_games} escritas"
    )
    os.replace(tmp_out, OUT_PATH)

    elapsed_total = time.time() - t0
    manifest = {
        "source_pgn": SOURCE_PGN,
        "n_games_indexed": n_games,
        "elapsed_seconds": round(elapsed_total, 2),
        "output_path": OUT_PATH,
        "method": (
            "Un pase secuencial en modo binario (f.readline()/f.tell() explicitos, "
            "sin iteracion implicita de linea) que registra byte_start/byte_end de "
            "cada partida y parsea SOLO las lineas de cabecera '[Clave \"valor\"]' via "
            "regex de texto plano. No se invoco python-chess ni se calculo ningun hash "
            "de identidad en este paso."
        ),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Indice completo: {n_games} partidas en {elapsed_total:.1f}s")
    print(f"Escrito en {OUT_PATH}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
