"""PASO 35 -- smoke test de ~100 posiciones REALES del Golden Dataset,
estratificadas (opening/middlegame/endgame/checks/captures/tactical/
repeated). Los casos sinteticos especificos (isolated pawn, pin, skewer,
fork, mate, repetition, etc.) ya estan cubiertos en
test_synthetic_positions.py -- este archivo valida que el pipeline
completo no rompe sobre datos REALES variados, sin tocar el Golden (solo
lectura)."""
import json
import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

from golden_reader import GoldenDataset  # noqa: E402
from position_features import compute_all_position_features  # noqa: E402


def _select_stratified_100(positions, seed=20260812):
    rng = np.random.default_rng(seed)
    eligible = positions[positions["human_move_id"] != -1]

    def sample(mask, k):
        pool = eligible[mask]
        k = min(k, len(pool))
        if k == 0:
            return pool.iloc[[]]
        idx = rng.choice(len(pool), size=k, replace=False)
        return pool.iloc[idx]

    parts = [
        sample(eligible["game_phase"] == "opening", 20),
        sample(eligible["game_phase"] == "middlegame", 20),
        sample(eligible["game_phase"] == "endgame", 20),
        sample(eligible["is_check"], 10),
        sample(eligible["move_is_capture"], 10),
        sample(eligible["tactical_or_critical"], 10),
        sample(eligible["position_occurrence_count"] > 1, 10),
    ]
    import pandas as pd
    combined = pd.concat(parts).drop_duplicates(subset=["row_id"])
    return combined


@pytest.fixture(scope="module")
def golden_sample():
    config = json.load(open(os.path.join(SCRIPTS_DIR, "..", "config", "enrichment_v3_config.json"), encoding="utf-8"))
    gd = GoldenDataset(config, os.path.join(SCRIPTS_DIR, "..", "config"))
    positions = gd.positions()
    sample = _select_stratified_100(positions)
    return config, sample


def test_smoke_sample_size_and_strata(golden_sample):
    _, sample = golden_sample
    assert 60 <= len(sample) <= 100  # puede haber solapamiento entre estratos (p.ej. check+capture)
    assert sample["row_id"].is_unique


def test_smoke_position_features_no_crash_and_sane_ranges(golden_sample):
    config, sample = golden_sample
    errors = []
    for _, r in sample.iterrows():
        try:
            feats = compute_all_position_features(r["fen"], config)
        except Exception as exc:
            errors.append((r["row_id"], type(exc).__name__, str(exc)))
            continue
        assert 0 <= feats["mobility_white"] <= 218  # cota superior conocida de movilidad en ajedrez
        assert 0 <= feats["mobility_black"] <= 218
        assert 0.0 <= feats["white_king_exposure_score"] <= 1.0
        assert 0.0 <= feats["black_king_exposure_score"] <= 1.0
        assert feats["hanging_material_white"] >= 0
        assert feats["hanging_material_black"] >= 0
    assert errors == [], f"{len(errors)} posiciones fallaron: {errors[:5]}"


def test_smoke_no_unexpected_nulls_in_key_fields(golden_sample):
    config, sample = golden_sample
    for _, r in sample.iterrows():
        feats = compute_all_position_features(r["fen"], config)
        for key in ("attacked_square_count_white", "attacked_square_count_black", "mobility_white",
                    "mobility_black", "white_isolated_pawns", "black_isolated_pawns"):
            assert feats[key] is not None, f"{key} nulo inesperado en row_id={r['row_id']}"
