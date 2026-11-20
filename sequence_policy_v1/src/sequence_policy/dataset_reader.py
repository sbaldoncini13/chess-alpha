"""Lectura del dataset de ventanas via memory mapping, sin duplicar los
tokens originales de tokenization_v1 y sin reconstruir tableros (seccion 7).

SequenceGameDataset: acceso a partidas completas (para auditorias/eval).
SequenceWindowDataset: acceso a ventanas individuales (para entrenamiento) --
lee el manifest de 04_build_sequence_dataset.py y arma cada input_ids/labels
al vuelo desde el memmap de moves_uint16.bin + special tokens.
"""
import os

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from sequence_policy.constants import BOS_STD_ID, EOS_ID, PAD_ID


class TokenMemmap:
    """Envoltorio fino sobre np.memmap del moves_uint16.bin de tokenization_v1
    -- read-only, un solo file handle reusado, nunca se copian los 81M tokens
    a una lista de Python."""

    def __init__(self, tokens_path: str):
        self.path = tokens_path
        self._mm = None

    @property
    def mm(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._mm

    def slice(self, offset: int, length: int) -> np.ndarray:
        return np.asarray(self.mm[offset:offset + length])


class SequenceWindowDataset(Dataset):
    """Una fila = una ventana. __getitem__ arma:
    input_ids: [S[context_start] .. S[context_end-1]]  (BOS/EOS insertados segun corresponda)
    labels:    [S[context_start+1] .. S[context_end]], con -100 fuera de [loss_start,loss_end]
    (indices en S-space: 0=BOS, 1..n=movimientos reales, n+1=EOS -- ver sequence_windows.py)
    """

    IGNORE_INDEX = -100

    def __init__(self, windows_parquet_path: str, tokens_path: str):
        table = pq.read_table(windows_parquet_path)
        self.windows = table.to_pydict()
        self.n = table.num_rows
        self.tokens = TokenMemmap(tokens_path)

    def __len__(self):
        return self.n

    def _s_index_to_token(self, s_idx: int, token_offset_start: int, game_length: int) -> int:
        if s_idx == 0:
            return BOS_STD_ID
        if s_idx == game_length + 1:
            return EOS_ID
        # s_idx in [1, game_length]: token real, posicion (s_idx-1) del array de la partida
        return int(self.tokens.slice(token_offset_start + s_idx - 1, 1)[0])

    def __getitem__(self, idx: int) -> dict:
        w = self.windows
        context_start = w["context_start"][idx]
        context_end = context_start + w["window_length"][idx]
        loss_start = w["loss_start"][idx]
        loss_end = w["loss_end"][idx]
        token_offset_start = w["token_offset_start"][idx]
        game_length = w["game_length"][idx]

        input_ids = [
            self._s_index_to_token(s, token_offset_start, game_length)
            for s in range(context_start, context_end)
        ]
        label_s_indices = range(context_start + 1, context_end + 1)
        labels = []
        for s in label_s_indices:
            if loss_start <= s <= loss_end:
                labels.append(self._s_index_to_token(s, token_offset_start, game_length))
            else:
                labels.append(self.IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
            "sample_id": w["sample_id"][idx],
            "bucket_id": w["bucket_id"][idx],
            "context_start": context_start,
        }


class SequenceGameDataset(Dataset):
    """Una fila = una partida COMPLETA (BOS + todos los movimientos + EOS),
    sin ventanas. Pensada para auditorias/evaluacion, no para entrenamiento
    de contexto acotado (una partida de 561 plies no entra en 256)."""

    def __init__(self, games_parquet_path: str, tokens_path: str):
        table = pq.read_table(games_parquet_path,
                               columns=["sample_id", "game_index", "token_offset", "ply_length"])
        self.games = table.to_pydict()
        self.n = table.num_rows
        self.tokens = TokenMemmap(tokens_path)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> dict:
        offset = self.games["token_offset"][idx]
        length = self.games["ply_length"][idx]
        moves = self.tokens.slice(offset, length).astype(np.int64).tolist()
        return {
            "sample_id": self.games["sample_id"][idx],
            "game_index": self.games["game_index"][idx],
            "input_ids": [BOS_STD_ID] + moves + [EOS_ID],
        }
