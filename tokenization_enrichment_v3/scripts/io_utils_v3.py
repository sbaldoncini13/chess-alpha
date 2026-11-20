"""Escritura atomica (tmp -> verificacion -> os.replace con reintento ante
PermissionError transitorio de Windows), mismo patron ya validado en
tokenization_v1/scripts/io_utils_v2.py (solo lectura, no importado -- se
reimplementa para que tokenization_enrichment_v3/ sea un modulo
completamente independiente, ni siquiera comparte codigo de IO con
tokenization_v1/)."""
import json
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPLACE_MAX_RETRIES = 5
_REPLACE_BACKOFF_SECONDS = 0.3


def _replace_with_retry(tmp_path: str, final_path: str):
    last_exc = None
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc


def atomic_write_json(final_path: str, obj: dict):
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_path, encoding="utf-8") as f:
        json.load(f)  # verificacion: debe poder releerse
    _replace_with_retry(tmp_path, final_path)


def atomic_write_parquet_df(final_path: str, df: pd.DataFrame):
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp_path = final_path + ".tmp"
    pq.write_table(table, tmp_path)
    check = pq.read_table(tmp_path, columns=[table.schema.names[0]])
    if check.num_rows != table.num_rows:
        raise IOError(f"Verificacion de parquet fallida: {final_path}")
    _replace_with_retry(tmp_path, final_path)


def atomic_write_bytes(final_path: str, data: bytes):
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_path, "rb") as f:
        check = f.read(min(len(data), 4096))
    if check != data[:len(check)]:
        raise IOError(f"Verificacion de escritura fallida: {final_path}")
    _replace_with_retry(tmp_path, final_path)
