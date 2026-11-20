"""Escritura atomica reusable - mismo patron ya usado sueltamente en
44/109/110/111/112, factorizado aca para los scripts de SEARCH TREE
DATASET V1.1 en vez de duplicarlo de nuevo en cada uno."""

import json
import os
from pathlib import Path

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    pd.read_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    with open(tmp, encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp, path)
