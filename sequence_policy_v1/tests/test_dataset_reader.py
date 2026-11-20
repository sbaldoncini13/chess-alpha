import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

from sequence_policy.constants import BOS_STD_ID, EOS_ID, PAD_ID
from sequence_policy.dataset_reader import SequenceWindowDataset
from sequence_policy.collator import SequenceBatchCollator


SEQ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Dos layouts posibles: repo principal (tokenization_v1/ hermano de
# sequence_policy_v1/) o bundle de Colab (copia propia en data/tokenized_dataset/,
# ver colab_bundle/tests/conftest.py). Se prueba el primero que exista, para
# no saltear estos tests solo por estar corriendo dentro del bundle.
_CANDIDATE_TOKEN_DIRS = [
    os.path.abspath(os.path.join(SEQ_ROOT, "..", "tokenization_v1", "outputs", "full_1000000")),
    os.path.abspath(os.path.join(SEQ_ROOT, "data", "tokenized_dataset")),
]
TOKENIZATION_OUT_DIR = next((p for p in _CANDIDATE_TOKEN_DIRS if os.path.isdir(p)), _CANDIDATE_TOKEN_DIRS[0])
WINDOWS_DEV = os.path.join(SEQ_ROOT, "data", "processed", "windows_dev.parquet")


def _skip_if_missing():
    if not (os.path.isfile(WINDOWS_DEV) and os.path.isdir(TOKENIZATION_OUT_DIR)):
        pytest.skip("requiere windows_dev.parquet y tokenization_v1/full_1000000 (Etapa C ya construida)")


def test_window_dataset_shapes_and_specials():
    _skip_if_missing()
    ds = SequenceWindowDataset(WINDOWS_DEV, os.path.join(TOKENIZATION_OUT_DIR, "moves_uint16.bin"))
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"]) == item["length"]
    assert item["input_ids"][0] in (BOS_STD_ID,) or item["input_ids"][0] != PAD_ID
    for tok in item["input_ids"]:
        assert tok != PAD_ID  # PAD nunca aparece en input_ids reales (solo lo agrega el collator)


def test_first_window_starts_with_bos():
    _skip_if_missing()
    ds = SequenceWindowDataset(WINDOWS_DEV, os.path.join(TOKENIZATION_OUT_DIR, "moves_uint16.bin"))
    w = ds.windows
    first_window_idx = next(i for i in range(len(w["context_start"])) if w["context_start"][i] == 0)
    item = ds[first_window_idx]
    assert item["input_ids"][0] == BOS_STD_ID


def test_labels_respect_loss_mask():
    _skip_if_missing()
    ds = SequenceWindowDataset(WINDOWS_DEV, os.path.join(TOKENIZATION_OUT_DIR, "moves_uint16.bin"))
    item = ds[0]
    n_real_labels = sum(1 for lb in item["labels"] if lb != SequenceWindowDataset.IGNORE_INDEX)
    w = ds.windows
    expected = w["loss_end"][0] - w["loss_start"][0] + 1
    assert n_real_labels == expected


def test_collator_dynamic_padding():
    _skip_if_missing()
    ds = SequenceWindowDataset(WINDOWS_DEV, os.path.join(TOKENIZATION_OUT_DIR, "moves_uint16.bin"))
    collator = SequenceBatchCollator()
    batch = [ds[i] for i in range(8)]
    out = collator(batch)
    max_len = max(item["length"] for item in batch)
    assert out["input_ids"].shape == (8, max_len)
    assert out["labels"].shape == (8, max_len)
    for i, item in enumerate(batch):
        n = item["length"]
        assert out["attention_mask"][i, :n].sum().item() == n
        assert out["attention_mask"][i, n:].sum().item() == 0
        if n < max_len:
            assert (out["input_ids"][i, n:] == PAD_ID).all()
            assert (out["labels"][i, n:] == SequenceWindowDataset.IGNORE_INDEX).all()
