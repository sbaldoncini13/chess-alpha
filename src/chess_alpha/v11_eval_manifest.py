"""Manifest de posiciones congelado, compartido por las Secciones 3/4/5 del
plan de evaluacion decisiva v1.1 (coverage/value ampliados + CPL/blunders).
Se genera UNA sola vez (seed fija) y se persiste - nunca se re-samplea en
corridas posteriores, para que los 4 checkpoints (y, dentro del mismo
manifest, coverage y CPL) se evaluen siempre sobre las mismas posiciones.

CPL (Seccion 5) reusa el mismo set de `coverage_and_cpl` que coverage
(Seccion 3), tal como pidio el usuario ("usar las mismas del benchmark
anterior si estan disponibles") - por eso hay un solo manifest con dos
listas, no dos manifests separados."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chess_alpha.config import LICHESS_TRAINING_DATASET_PATH, V11_EVAL_MANIFEST_PATH

SEED = 42
N_COVERAGE_AND_CPL = 500
N_VALUE = 200


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[idx]
    return [{"fen": row.fen, "result": row.result} for row in sub.itertuples(index=False)]


def build_or_load_manifest(seed: int = SEED, n_coverage_and_cpl: int = N_COVERAGE_AND_CPL,
                            n_value: int = N_VALUE, manifest_path: Path = None) -> dict:
    """manifest_path: override opcional (default None = V11_EVAL_MANIFEST_PATH, el
    manifest compartido de 500/200 usado por los 4 checkpoints v1.1 - NUNCA
    se re-samplea con parametros distintos). Pasar una ruta nueva para
    generar/cargar un manifest v2 independiente (ej. N=1000/300 para el
    gate arquitectonico Epoch4) sin tocar el original."""
    path = manifest_path or V11_EVAL_MANIFEST_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["seed"] != seed or manifest["n_coverage_and_cpl"] != n_coverage_and_cpl \
                or manifest["n_value"] != n_value:
            raise ValueError(
                f"El manifest existente en {path} fue generado con "
                f"seed={manifest['seed']}/n_cov={manifest['n_coverage_and_cpl']}/n_val={manifest['n_value']}, "
                f"distinto a lo pedido ahora (seed={seed}/n_cov={n_coverage_and_cpl}/n_val={n_value}). "
                f"No se re-samplea automaticamente - generar un manifest v2 en otra ruta (manifest_path=...), "
                f"sin borrar este."
            )
        return manifest

    df = pd.read_parquet(LICHESS_TRAINING_DATASET_PATH, columns=["fen", "result"])
    coverage_and_cpl = _sample_rows(df, n_coverage_and_cpl, seed)
    value = _sample_rows(df, n_value, seed)  # mismo seed, distinto n -> muestra independiente de coverage

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_coverage_and_cpl": n_coverage_and_cpl,
        "n_value": n_value,
        "n_source_positions": len(df),
        "source_dataset": str(LICHESS_TRAINING_DATASET_PATH),
        "coverage_and_cpl": coverage_and_cpl,
        "value": value,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return manifest


def manifest_fingerprint(manifest: dict) -> str:
    """Hash corto para imprimir al arrancar cada corrida de checkpoint y
    confirmar visualmente que las 4 corridas cargaron el mismo manifest."""
    import hashlib
    h = hashlib.sha256()
    for row in manifest["coverage_and_cpl"] + manifest["value"]:
        h.update(row["fen"].encode("utf-8"))
    return h.hexdigest()[:16]


if __name__ == "__main__":
    m = build_or_load_manifest()
    print(f"Manifest en {V11_EVAL_MANIFEST_PATH}")
    print(f"  coverage_and_cpl: {len(m['coverage_and_cpl'])} posiciones")
    print(f"  value: {len(m['value'])} posiciones")
    print(f"  fingerprint: {manifest_fingerprint(m)}")
