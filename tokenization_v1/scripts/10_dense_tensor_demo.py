"""PASO 6 (validacion final): materializa el tensor denso (23,8,8) float32
SOLO para una muestra pequena (config.materialize_dense_tensors.demo_sample_size)
de las posiciones ya escritas en shards_v2/<test_name>/, para demostrar
shape/dtype/orientacion sobre datos reales sin repetir el computo sobre las
~818k posiciones completas del piloto (ver justificacion en
tokenization_v2_config.json y SCHEMA_V2.md).

Solo lee shards_v2/<test_name>/ (FEN por posicion) -- no toca v1 ni re-parsea
nada. Escribe outputs_v2/<test_name>/dense_tensor_demo.npz.
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
from position_state import board_to_tensor_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    base = os.path.dirname(args.config)
    shard_dir = os.path.abspath(os.path.join(base, config["shard_dir"], args.test_name))
    out_dir = os.path.abspath(os.path.join(base, config["output_dir"], args.test_name))
    sample_size = config["materialize_dense_tensors"]["demo_sample_size"]

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".positions.parquet"))
    tables = [pq.read_table(os.path.join(shard_dir, f), columns=["row_id", "fen"]) for f in shard_files]
    full = pa.concat_tables(tables).to_pandas().sort_values("row_id").reset_index(drop=True)

    sample = full.iloc[:sample_size] if len(full) >= sample_size else full

    t0 = time.time()
    tensors = np.zeros((len(sample), 23, 8, 8), dtype=np.float32)
    for i, fen in enumerate(sample["fen"]):
        tensors[i] = board_to_tensor_v2(chess.Board(fen))
    elapsed = time.time() - t0

    out_path = os.path.join(out_dir, "dense_tensor_demo.npz")
    np.savez_compressed(out_path, tensors=tensors, row_ids=sample["row_id"].to_numpy())

    result = {
        "test_name": args.test_name, "n_positions_materializadas": len(sample),
        "shape": list(tensors.shape), "dtype": str(tensors.dtype),
        "elapsed_seconds": round(elapsed, 3),
        "positions_per_second": round(len(sample) / elapsed, 1) if elapsed > 0 else None,
        "bytes_per_position": 23 * 8 * 8 * 4,
        "output_path": out_path,
        "proyeccion_full_pilot_gb": round(len(full) * 23 * 8 * 8 * 4 / (1024 ** 3), 3),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
