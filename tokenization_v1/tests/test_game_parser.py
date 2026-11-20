import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from game_parser import (
    INVALID_ILLEGAL_MOVE,
    INVALID_RESULT_MISSING,
    INVALID_VARIANT,
    VALID,
    compute_identity_hash,
    parse_and_validate_game,
)


def write_pgn(tmp_path, name, text):
    path = tmp_path / name
    # utf-8-sig en el primer juego del archivo, igual que el dataset real
    path.write_bytes(text.encode("utf-8-sig"))
    return str(path), 0, len(text.encode("utf-8-sig"))


NORMAL_GAME = """[Event "Test"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2250"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""

FEN_GAME = """[Event "Test FEN"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[SetUp "1"]
[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]

1. a8=Q+ Kd7 1-0
"""

TRUNCATED_GAME = """[Event "Test truncated"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *
"""

VARIANT_GAME = """[Event "Test variant"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[Variant "Chess960"]

1. e4 e5 1-0
"""


def test_valid_normal_game(tmp_path):
    path, start, end = write_pgn(tmp_path, "g1.pgn", NORMAL_GAME)
    r = parse_and_validate_game(path, start, end, draw_rank=0, game_index=0)
    assert r["status"] == VALID
    assert r["ply_count"] == 10
    assert r["result_code"] == 1
    assert len(r["action_ids"]) == 10
    assert r["identity_hash"] is not None


def test_valid_game_from_fen(tmp_path):
    path, start, end = write_pgn(tmp_path, "g2.pgn", FEN_GAME)
    r = parse_and_validate_game(path, start, end, draw_rank=1, game_index=1)
    assert r["status"] == VALID
    assert r["initial_fen"].startswith("4k3/P7/8/8/8/8/8/4K3")
    assert r["ply_count"] == 2


def test_illegal_move_detected(tmp_path):
    bad_game = NORMAL_GAME.replace("5. O-O Be7", "5. O-O Qxh7")
    path, start, end = write_pgn(tmp_path, "g4.pgn", bad_game)
    r = parse_and_validate_game(path, start, end, draw_rank=3, game_index=3)
    assert r["status"] == INVALID_ILLEGAL_MOVE
    assert r["error_move"] == "Qxh7"
    assert r["error_ply"] == 9
    assert r["error_fen"] is not None


def test_truncated_game(tmp_path):
    path, start, end = write_pgn(tmp_path, "g5.pgn", TRUNCATED_GAME)
    r = parse_and_validate_game(path, start, end, draw_rank=4, game_index=4)
    assert r["status"] == INVALID_RESULT_MISSING


def test_unsupported_variant(tmp_path):
    path, start, end = write_pgn(tmp_path, "g6.pgn", VARIANT_GAME)
    r = parse_and_validate_game(path, start, end, draw_rank=5, game_index=5)
    assert r["status"] == INVALID_VARIANT


def test_identity_hash_deterministic():
    h1 = compute_identity_hash("standard", "start", ["e2e4", "e7e5"])
    h2 = compute_identity_hash("standard", "start", ["e2e4", "e7e5"])
    h3 = compute_identity_hash("standard", "start", ["e2e4", "d7d5"])
    assert h1 == h2
    assert h1 != h3
