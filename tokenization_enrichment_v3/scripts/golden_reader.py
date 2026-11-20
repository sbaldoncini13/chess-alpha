"""Acceso SOLO LECTURA al Golden Dataset `chess_alpha_tokenization_golden_
test10000_v1`. Nunca escribe en tokenization_v1/. Todas las funciones de
este modulo son de lectura pura -- ni un solo os.replace/write hacia el
arbol del Golden.

Provee ademas un indice por partida (game_id -> row_id de inicio + cantidad
de filas) que permite navegacion O(1) hacia atras/adelante DENTRO de una
partida (usado por sequences.py, eval_trajectory.py, terminal_distance.py)
sin volver a tocar los arrays crudos de tokv1.0 -- el propio positions.parquet
del Golden ya trae game_id/ply/human_move_id por fila, eso alcanza.
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class GoldenDataset:
    def __init__(self, config: dict, config_dir: str):
        src = config["source_golden"]
        self.shards_dir = os.path.abspath(os.path.join(config_dir, src["golden_shards_dir"]))
        self.engine_dir = os.path.abspath(os.path.join(config_dir, src["golden_engine_dir"]))
        self.golden_dataset_id = src["golden_dataset_id"]
        self.golden_dataset_version = src["golden_dataset_version"]

        self._positions = None
        self._engine_eval = None
        self._engine_topk = None
        self._engine_policy = None
        self._engine_crit = None
        self._game_index = None

    def positions(self) -> pd.DataFrame:
        if self._positions is None:
            files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".positions.parquet"))
            tables = [pq.read_table(os.path.join(self.shards_dir, f)) for f in files]
            self._positions = pa.concat_tables(tables).to_pandas().sort_values("row_id").reset_index(drop=True)
        return self._positions

    def _engine_shard_tags(self):
        return sorted(
            d for d in os.listdir(self.engine_dir)
            if os.path.isdir(os.path.join(self.engine_dir, d)) and d.startswith("shard_")
        )

    def engine_eval(self) -> pd.DataFrame:
        if self._engine_eval is None:
            tags = self._engine_shard_tags()
            tables = [pq.read_table(os.path.join(self.engine_dir, t, "engine_eval.parquet")) for t in tags]
            self._engine_eval = pa.concat_tables(tables).to_pandas().sort_values("row_id").reset_index(drop=True)
        return self._engine_eval

    def engine_topk(self) -> pd.DataFrame:
        if self._engine_topk is None:
            tags = self._engine_shard_tags()
            tables = [pq.read_table(os.path.join(self.engine_dir, t, "engine_topk.parquet")) for t in tags]
            self._engine_topk = pa.concat_tables(tables).to_pandas()
        return self._engine_topk

    def engine_policy(self) -> pd.DataFrame:
        if self._engine_policy is None:
            tags = self._engine_shard_tags()
            tables = [pq.read_table(os.path.join(self.engine_dir, t, "engine_policy_target.parquet")) for t in tags]
            self._engine_policy = pa.concat_tables(tables).to_pandas()
        return self._engine_policy

    def engine_criticality(self) -> pd.DataFrame:
        if self._engine_crit is None:
            tags = self._engine_shard_tags()
            tables = [pq.read_table(os.path.join(self.engine_dir, t, "engine_criticality.parquet")) for t in tags]
            self._engine_crit = pa.concat_tables(tables).to_pandas()
        return self._engine_crit

    def game_index(self) -> pd.DataFrame:
        """Una fila por game_id: row_id_start, n_rows (=ply_count+1), row_id_end.
        Valido porque el pipeline v2 asigna row_id de forma contigua por
        partida en orden de ply (invariante ya verificado por el validador
        del Golden -- filas_por_partida == ply_count+1, row_id contiguo)."""
        if self._game_index is None:
            pos = self.positions()[["row_id", "game_id", "ply"]]
            grp = pos.groupby("game_id").agg(row_id_start=("row_id", "min"), row_id_end=("row_id", "max"),
                                              n_rows=("row_id", "size"))
            grp = grp.reset_index()
            self._game_index = grp.set_index("game_id")
        return self._game_index

    def base_shard_tags(self):
        return sorted({f.split(".")[0] for f in os.listdir(self.shards_dir) if f.endswith(".positions.parquet")})

    def positions_for_shard(self, tag: str) -> pd.DataFrame:
        t = pq.read_table(os.path.join(self.shards_dir, f"{tag}.positions.parquet"))
        return t.to_pandas().sort_values("row_id").reset_index(drop=True)
