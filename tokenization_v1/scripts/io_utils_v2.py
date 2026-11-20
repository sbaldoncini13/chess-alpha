"""Escritura atomica (tmp -> verificacion -> os.replace), igual patron que
04_run_tokenization.py (tokv1.0). Reimplementado aca (no importado desde un
script cuyo nombre empieza con un digito) para mantener tokenization_v1/
modulos limpiamente importables entre si."""
import io
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPLACE_MAX_RETRIES = 5
_REPLACE_BACKOFF_SECONDS = 0.3


def _replace_with_retry(tmp_path: str, final_path: str):
    """os.replace() puede fallar transitoriamente en Windows con
    PermissionError/WinError 32 (\"el proceso no tiene acceso al archivo
    porque esta siendo utilizado por otro proceso\") cuando un antivirus/
    indexador abre brevemente el archivo recien escrito. Encontrado en
    produccion durante el full enrichment (corrida de horas) -- una
    posicion problematica de infraestructura transitoria no debe tirar
    abajo una corrida larga, igual criterio que los reintentos de
    posiciones individuales de Stockfish. Reintenta unicamente
    PermissionError; cualquier otro error se propaga de inmediato."""
    last_exc = None
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc


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


def atomic_write_json(final_path: str, obj: dict):
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_path, encoding="utf-8") as f:
        reloaded = json.load(f)
    if reloaded != obj:
        raise IOError(f"Verificacion de escritura JSON fallida: {final_path}")
    _replace_with_retry(tmp_path, final_path)


def atomic_write_parquet(final_path: str, table: pa.Table):
    tmp_path = final_path + ".tmp"
    pq.write_table(table, tmp_path)
    check = pq.read_table(tmp_path, columns=[table.schema.names[0]])
    if check.num_rows != table.num_rows:
        raise IOError(f"Verificacion de parquet fallida: {final_path}")
    _replace_with_retry(tmp_path, final_path)


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def atomic_write_npy(final_path: str, arr: np.ndarray):
    atomic_write_bytes(final_path, npy_bytes(arr))
