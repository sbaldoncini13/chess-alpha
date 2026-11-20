import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from engine_cache import ConfigMismatchError, EngineCache, compute_request_key  # noqa: E402

BASE_META = {
    "engine_version": "Stockfish 18", "engine_binary_sha256": "abc123",
    "depth": 10, "multipv_root": 3, "multipv_child": 1, "threads": 1, "hash_mb": 16,
}


def test_request_key_deterministic():
    a = compute_request_key("fen1", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    b = compute_request_key("fen1", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    assert a == b


def test_request_key_differs_on_fen():
    a = compute_request_key("fen1", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    b = compute_request_key("fen2", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    assert a != b


def test_request_key_differs_on_request_type():
    root = compute_request_key("fen1", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    child = compute_request_key("fen1", "Stockfish 18", "abc", 10, 1, 1, 16, "child_single_pv")
    assert root != child


def test_request_key_differs_on_depth_multipv_engine_version():
    base = compute_request_key("fen1", "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    diff_depth = compute_request_key("fen1", "Stockfish 18", "abc", 12, 3, 1, 16, "root_multipv")
    diff_multipv = compute_request_key("fen1", "Stockfish 18", "abc", 10, 5, 1, 16, "root_multipv")
    diff_engine = compute_request_key("fen1", "Stockfish 17", "abc", 10, 3, 1, 16, "root_multipv")
    diff_binary = compute_request_key("fen1", "Stockfish 18", "xyz", 10, 3, 1, 16, "root_multipv")
    assert len({base, diff_depth, diff_multipv, diff_engine, diff_binary}) == 5


def test_request_key_differs_on_halfmove_clock_only():
    # dos FEN que solo difieren en halfmove_clock deben tener claves distintas
    fen_a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    fen_b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 5 1"
    a = compute_request_key(fen_a, "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    b = compute_request_key(fen_b, "Stockfish 18", "abc", 10, 3, 1, 16, "root_multipv")
    assert a != b


def test_cache_put_get_root_round_trip(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    with EngineCache(db_path, BASE_META) as cache:
        key = compute_request_key("fen1", "Stockfish 18", "abc123", 10, 3, 1, 16, "root_multipv")
        assert cache.get_root(key) is None
        payload = {"best_move_id": 1, "best_move_uci": "e2e4", "topk": [{"rank": 1, "score_cp": 40}]}
        cache.put_root(key, "fen1", payload)
        assert cache.get_root(key) == payload


def test_cache_hit_miss_counters(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    with EngineCache(db_path, BASE_META) as cache:
        key = compute_request_key("fen1", "Stockfish 18", "abc123", 10, 3, 1, 16, "root_multipv")
        cache.get_root(key)  # miss
        cache.put_root(key, "fen1", {"best_move_id": 1, "best_move_uci": "e2e4", "topk": []})
        cache.get_root(key)  # hit
        cache.get_root(key)  # hit
        counts = cache.counts()
        assert counts["root_misses"] == 1
        assert counts["root_hits"] == 2
        assert counts["n_root_entries"] == 1


def test_cache_persists_across_reopen_same_config(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    key = compute_request_key("fen1", "Stockfish 18", "abc123", 10, 3, 1, 16, "root_multipv")
    with EngineCache(db_path, BASE_META) as cache:
        cache.put_root(key, "fen1", {"best_move_id": 1, "best_move_uci": "e2e4", "topk": []})
    # simula reinicio: se abre un NUEVO EngineCache sobre el mismo archivo
    with EngineCache(db_path, BASE_META) as cache2:
        assert cache2.get_root(key) is not None


def test_cache_rejects_mismatched_config_on_reopen(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    with EngineCache(db_path, BASE_META):
        pass
    changed_meta = dict(BASE_META)
    changed_meta["depth"] = 12
    with pytest.raises(ConfigMismatchError):
        EngineCache(db_path, changed_meta)


def test_cache_child_round_trip(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    with EngineCache(db_path, BASE_META) as cache:
        key = compute_request_key("fen_child", "Stockfish 18", "abc123", 10, 1, 1, 16, "child_single_pv")
        assert cache.get_child(key) is None
        cache.put_child(key, "fen_child", {"eval_after_stm": -35})
        assert cache.get_child(key) == {"eval_after_stm": -35}


def test_cache_put_is_idempotent_does_not_overwrite(tmp_path):
    db_path = str(tmp_path / "cache.sqlite")
    with EngineCache(db_path, BASE_META) as cache:
        key = compute_request_key("fen1", "Stockfish 18", "abc123", 10, 3, 1, 16, "root_multipv")
        cache.put_root(key, "fen1", {"best_move_id": 1, "best_move_uci": "e2e4", "topk": []})
        cache.put_root(key, "fen1", {"best_move_id": 999, "best_move_uci": "SHOULD_NOT_OVERWRITE", "topk": []})
        assert cache.get_root(key)["best_move_uci"] == "e2e4"
