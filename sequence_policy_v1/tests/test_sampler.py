import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from sequence_policy.sampler import DeterministicWindowSampler


def _lengths(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(10, 256, size=n)


def test_same_seed_same_first_1000_indices():
    lengths = _lengths()
    s1 = DeterministicWindowSampler(lengths, batch_size=8, seed=42)
    s2 = DeterministicWindowSampler(lengths, batch_size=8, seed=42)
    flat1 = [i for b in s1 for i in b][:1000]
    flat2 = [i for b in s2 for i in b][:1000]
    assert flat1 == flat2


def test_different_seed_different_order():
    lengths = _lengths()
    s1 = DeterministicWindowSampler(lengths, batch_size=8, seed=1)
    s2 = DeterministicWindowSampler(lengths, batch_size=8, seed=2)
    flat1 = [i for b in s1 for i in b][:1000]
    flat2 = [i for b in s2 for i in b][:1000]
    assert flat1 != flat2


def test_resume_mid_reproduces_next_batch():
    lengths = _lengths()
    s_full = DeterministicWindowSampler(lengths, batch_size=8, seed=7)
    all_batches = list(s_full)

    s_partial = DeterministicWindowSampler(lengths, batch_size=8, seed=7)
    it = iter(s_partial)
    for _ in range(10):
        next(it)
    state = s_partial.state_dict()

    s_resumed = DeterministicWindowSampler(lengths, batch_size=8, seed=7)
    s_resumed.load_state_dict(state)
    next_batch_resumed = next(iter(s_resumed))

    assert next_batch_resumed == all_batches[10]


def test_train_dev_never_mixed():
    # El sampler opera sobre UN solo dataset (train O dev) por construccion;
    # esta prueba confirma que no hay mecanismo que mezcle indices de otro
    # rango -- todos los indices producidos estan en [0, n).
    lengths = _lengths(n=200)
    s = DeterministicWindowSampler(lengths, batch_size=8, seed=3)
    all_idx = [i for b in s for i in b]
    assert min(all_idx) >= 0
    assert max(all_idx) < 200
    assert len(set(all_idx)) == 200  # cada indice aparece exactamente una vez


def test_no_custom_fen_games_in_v1():
    import json
    seq_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reserve_path = os.path.join(seq_root, "data", "manifests", "custom_fen_reserve.json")
    split_path_train = os.path.join(seq_root, "data", "splits", "train_games.parquet")
    if not (os.path.isfile(reserve_path) and os.path.isfile(split_path_train)):
        import pytest
        pytest.skip("requiere Etapa A completa")
    import pyarrow.parquet as pq
    reserve = json.load(open(reserve_path, encoding="utf-8"))
    excluded_sids = {g["sample_id"] for g in reserve["games"]}
    train_sids = set(pq.read_table(split_path_train, columns=["sample_id"]).column("sample_id").to_pylist())
    assert len(train_sids & excluded_sids) == 0
