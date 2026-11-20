import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from history import V1MovesReader  # noqa: E402
from tokenizer import move_to_action_id  # noqa: E402


@pytest.fixture
def tiny_v1_outputs(tmp_path):
    games_sans = [["e4", "e5", "Nf3", "Nc6", "Bb5"], ["d4", "d5"]]
    all_ids = []
    offsets = []
    lengths = []
    offset = 0
    for sans in games_sans:
        board = chess.Board()
        ids = []
        for san in sans:
            move = board.parse_san(san)
            ids.append(move_to_action_id(move))
            board.push(move)
        all_ids.extend(ids)
        offsets.append(offset)
        lengths.append(len(ids))
        offset += len(ids)
    np.array(all_ids, dtype=np.uint16).tofile(str(tmp_path / "moves_uint16.bin"))
    np.save(str(tmp_path / "game_offsets.npy"), np.array(offsets, dtype=np.int64))
    np.save(str(tmp_path / "game_lengths.npy"), np.array(lengths, dtype=np.uint16))
    return str(tmp_path), games_sans


def test_get_move_history_o1_reference_no_replay_needed(tiny_v1_outputs):
    v1_dir, games_sans = tiny_v1_outputs
    reader = V1MovesReader(v1_dir)

    board = chess.Board()
    expected_ids = []
    for san in games_sans[0]:
        move = board.parse_san(san)
        expected_ids.append(move_to_action_id(move))
        board.push(move)

    # k=4 en ply=5 (toda la partida tiene 5 plies) -> ultimos 4
    hist = reader.get_move_history(game_id=0, ply=5, k=4)
    assert list(hist) == expected_ids[1:5]

    # k mayor que ply disponible: devuelve todo lo que hay, sin rellenar
    hist_all = reader.get_move_history(game_id=0, ply=5, k=32)
    assert list(hist_all) == expected_ids

    # ply=0: sin historial
    hist_empty = reader.get_move_history(game_id=0, ply=0, k=8)
    assert len(hist_empty) == 0


def test_get_move_history_second_game_independent(tiny_v1_outputs):
    v1_dir, games_sans = tiny_v1_outputs
    reader = V1MovesReader(v1_dir)
    hist = reader.get_move_history(game_id=1, ply=2, k=4)
    assert len(hist) == 2
