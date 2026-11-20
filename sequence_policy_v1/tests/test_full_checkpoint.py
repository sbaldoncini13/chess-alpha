"""Pruebas del schema de checkpoint del full training (full_checkpoint.py):
warm-start SOLO de pesos, deteccion de arquitectura incompatible, deteccion
de NaN/Inf en el checkpoint del piloto, rechazo de checkpoint con schema
incorrecto, y que optimizer/scheduler/sampler nuevos NO se confunden con
una reanudacion del piloto."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.checkpointing import save_checkpoint
from sequence_policy.full_checkpoint import load_full_checkpoint, save_full_checkpoint, warm_start_from_pilot
from sequence_policy.model import CausalMoveTransformer
from sequence_policy.sampler import DeterministicWindowSampler

KW = dict(vocab_size=20483, d_model=32, num_layers=2, num_heads=2, head_dim=16,
          ffn_hidden=64, context_length=16, dropout=0.0, attention_dropout=0.0)
KW_DIFFERENT = dict(KW, d_model=64)


def _make_pilot_checkpoint(tmp_path, corrupt_nan=False):
    m = CausalMoveTransformer(**KW)
    if corrupt_nan:
        with torch.no_grad():
            list(m.parameters())[0].fill_(float("nan"))
    path = str(tmp_path / "pilot.ckpt")
    save_checkpoint(path, m, global_step=7532, tokens_seen=5000327, config={"model": KW})
    return path


def test_warm_start_loads_weights_only(tmp_path):
    pilot_path = _make_pilot_checkpoint(tmp_path)
    m_full = CausalMoveTransformer(**KW)
    prov = warm_start_from_pilot(pilot_path, m_full, expected_pilot_model_config=KW)
    assert prov["pilot_global_step_reference"] == 7532
    assert prov["pilot_tokens_seen_reference"] == 5000327
    assert len(prov["pilot_checkpoint_sha256"]) == 64


def test_warm_start_rejects_architecture_mismatch(tmp_path):
    pilot_path = _make_pilot_checkpoint(tmp_path)
    m_full = CausalMoveTransformer(**KW_DIFFERENT)
    with pytest.raises(ValueError, match="Arquitectura"):
        warm_start_from_pilot(pilot_path, m_full, expected_pilot_model_config=KW_DIFFERENT)


def test_warm_start_rejects_nan_weights(tmp_path):
    pilot_path = _make_pilot_checkpoint(tmp_path, corrupt_nan=True)
    m_full = CausalMoveTransformer(**KW)
    with pytest.raises(ValueError, match="NaN"):
        warm_start_from_pilot(pilot_path, m_full, expected_pilot_model_config=KW)


def test_warm_start_rejects_missing_file(tmp_path):
    m_full = CausalMoveTransformer(**KW)
    with pytest.raises(FileNotFoundError):
        warm_start_from_pilot(str(tmp_path / "no_existe.ckpt"), m_full)


def test_full_checkpoint_counters_are_distinct(tmp_path):
    """run_tokens_seen y cumulative_tokens_seen_including_pilot NUNCA deben
    mezclarse como si fueran el mismo contador."""
    pilot_path = _make_pilot_checkpoint(tmp_path)
    m_full = CausalMoveTransformer(**KW)
    prov = warm_start_from_pilot(pilot_path, m_full, expected_pilot_model_config=KW)
    opt = torch.optim.AdamW(m_full.parameters(), lr=1e-4)
    sampler = DeterministicWindowSampler(np.array([50] * 50), batch_size=8, seed=1)

    full_path = str(tmp_path / "full_last.ckpt")
    save_full_checkpoint(
        full_path, m_full, opt, None, None, sampler,
        run_global_step=3, run_tokens_seen=999,
        epoch_equivalent=0.0001, model_config=KW,
        dataset_hashes={"d": "x"}, split_hashes={"s": "y"},
        pilot_checkpoint_sha256=prov["pilot_checkpoint_sha256"],
        pilot_global_step_reference=prov["pilot_global_step_reference"],
        pilot_tokens_seen_reference=prov["pilot_tokens_seen_reference"],
        warm_start_source=prov["warm_start_source"], code_version="test",
    )
    m_reload = CausalMoveTransformer(**KW)
    ckpt = load_full_checkpoint(full_path, m_reload, expected_model_config=KW)
    extra = ckpt["extra"]
    assert extra["run_tokens_seen"] == 999
    assert extra["pilot_tokens_seen_reference"] == 5000327
    assert extra["cumulative_tokens_seen_including_pilot"] == 999 + 5000327
    assert extra["run_tokens_seen"] != extra["cumulative_tokens_seen_including_pilot"]
    assert extra["classification"] == "WEIGHTS_WARM_START_FROM_PILOT_FULL_NEW_EPOCH"


def test_load_full_checkpoint_rejects_pilot_schema(tmp_path):
    """Un checkpoint del PILOTO (schema viejo, sin 'extra.schema_version')
    cargado por error con load_full_checkpoint debe fallar explicitamente,
    no interpretarse en silencio como un checkpoint de full training."""
    pilot_path = _make_pilot_checkpoint(tmp_path)
    m = CausalMoveTransformer(**KW)
    with pytest.raises(ValueError, match="schema"):
        load_full_checkpoint(pilot_path, m, expected_model_config=KW)


def test_rng_states_are_saved_and_restorable(tmp_path):
    pilot_path = _make_pilot_checkpoint(tmp_path)
    m_full = CausalMoveTransformer(**KW)
    prov = warm_start_from_pilot(pilot_path, m_full, expected_pilot_model_config=KW)
    opt = torch.optim.AdamW(m_full.parameters(), lr=1e-4)
    sampler = DeterministicWindowSampler(np.array([50] * 50), batch_size=8, seed=1)
    full_path = str(tmp_path / "full_last.ckpt")
    torch.manual_seed(123)
    save_full_checkpoint(
        full_path, m_full, opt, None, None, sampler, run_global_step=1, run_tokens_seen=1,
        epoch_equivalent=0.0, model_config=KW, dataset_hashes=None, split_hashes=None,
        pilot_checkpoint_sha256=prov["pilot_checkpoint_sha256"],
        pilot_global_step_reference=prov["pilot_global_step_reference"],
        pilot_tokens_seen_reference=prov["pilot_tokens_seen_reference"],
        warm_start_source=prov["warm_start_source"], code_version="test",
    )
    m_reload = CausalMoveTransformer(**KW)
    ckpt = load_full_checkpoint(full_path, m_reload, expected_model_config=KW)
    assert ckpt["rng_torch_cpu"] is not None
    assert ckpt["rng_python"] is not None
    assert ckpt["rng_numpy"] is not None
