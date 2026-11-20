"""Historial de movimientos previos (PASO 19): referencia compacta O(1) a los
arrays YA consolidados de tokv1.0 (moves_uint16.bin + game_offsets.npy), sin
duplicar los action_id de cada partida en el dataset v2. No se toca ni se
reescribe ningun archivo de v1 -- solo lectura via memmap.
"""
import os

import numpy as np


class V1MovesReader:
    """Envoltorio de solo lectura sobre outputs/<test_name>/ de tokv1.0."""

    def __init__(self, v1_output_dir: str):
        self.moves = np.memmap(
            os.path.join(v1_output_dir, "moves_uint16.bin"), dtype=np.uint16, mode="r"
        )
        self.game_offsets = np.load(os.path.join(v1_output_dir, "game_offsets.npy"))
        self.game_lengths = np.load(os.path.join(v1_output_dir, "game_lengths.npy"))

    def game_action_ids(self, game_id: int) -> np.ndarray:
        start = int(self.game_offsets[game_id])
        length = int(self.game_lengths[game_id])
        return self.moves[start:start + length]

    def get_move_history(self, game_id: int, ply: int, k: int) -> np.ndarray:
        """Ultimos `k` action_id jugados ANTES de `ply` (0-based) en la
        partida `game_id`. Si `ply < k`, devuelve menos de `k` elementos
        (nunca rellena con valores inventados). `ply=0` devuelve un array
        vacio (posicion inicial, sin historial)."""
        game_moves = self.game_action_ids(game_id)
        lo = max(0, ply - k)
        return np.array(game_moves[lo:ply])
