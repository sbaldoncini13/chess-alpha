import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from sample_selection import select_stratified_sample  # noqa: E402


def _synthetic_pool(n=500, seed=0):
    rng = np.random.default_rng(seed)
    phases = rng.choice(["opening", "middlegame", "endgame"], size=n)
    is_check = rng.random(n) < 0.1
    tactical = rng.random(n) < 0.1
    return pd.DataFrame({
        "row_id": np.arange(n), "game_phase": phases,
        "is_check": is_check, "tactical_or_critical": tactical,
    })


def test_sample_size_does_not_exceed_n_total():
    df = _synthetic_pool(500)
    ids = select_stratified_sample(df, n_total=100, seed=1, n_check_quota=10, n_tactical_quota=10)
    assert len(ids) <= 100


def test_sample_no_duplicates():
    df = _synthetic_pool(500)
    ids = select_stratified_sample(df, n_total=200, seed=1, n_check_quota=20, n_tactical_quota=20)
    assert len(ids) == len(set(ids.tolist()))


def test_sample_is_sorted():
    df = _synthetic_pool(500)
    ids = select_stratified_sample(df, n_total=100, seed=1, n_check_quota=10, n_tactical_quota=10)
    assert list(ids) == sorted(ids.tolist())


def test_sample_deterministic_same_seed():
    df = _synthetic_pool(500)
    a = select_stratified_sample(df, n_total=150, seed=42, n_check_quota=15, n_tactical_quota=15)
    b = select_stratified_sample(df, n_total=150, seed=42, n_check_quota=15, n_tactical_quota=15)
    assert list(a) == list(b)


def test_sample_different_seed_gives_different_sample():
    df = _synthetic_pool(500)
    a = select_stratified_sample(df, n_total=150, seed=1, n_check_quota=15, n_tactical_quota=15)
    b = select_stratified_sample(df, n_total=150, seed=2, n_check_quota=15, n_tactical_quota=15)
    assert list(a) != list(b)


def test_sample_includes_check_positions_when_available():
    df = _synthetic_pool(500)
    ids = select_stratified_sample(df, n_total=100, seed=1, n_check_quota=20, n_tactical_quota=0)
    selected = df[df["row_id"].isin(ids)]
    assert selected["is_check"].sum() >= min(20, df["is_check"].sum())


def test_sample_covers_all_three_phases_when_available():
    df = _synthetic_pool(500)
    ids = select_stratified_sample(df, n_total=300, seed=1, n_check_quota=0, n_tactical_quota=0)
    selected = df[df["row_id"].isin(ids)]
    assert set(selected["game_phase"].unique()) == {"opening", "middlegame", "endgame"}


def test_sample_handles_small_pool_gracefully():
    df = _synthetic_pool(20)
    ids = select_stratified_sample(df, n_total=1000, seed=1, n_check_quota=150, n_tactical_quota=100)
    assert len(ids) <= 20
    assert len(ids) == len(set(ids.tolist()))
