"""Prueba la correccion de run_training_step: el dtype de AMP (fp16 vs bf16)
debe estar desacoplado de la presencia de un GradScaler, no asumido siempre
bf16 como antes. Sin GPU real, fp16+GradScaler CUDA queda marcado
GPU_RUNTIME_VALIDATION_PENDING_COLAB -- aca solo se valida en CPU lo que es
genuinamente validable en CPU (bf16 autocast SI funciona en CPU; fp16
autocast no esta soportado en CPU por PyTorch, por eso no se prueba numerico
aca, solo la logica de validacion de argumentos)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.model import CausalMoveTransformer
from sequence_policy.trainer import run_training_step

VOCAB = 20483
KW = dict(vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, head_dim=16,
          ffn_hidden=128, context_length=32, dropout=0.0, attention_dropout=0.0)


def make_batch(bsz=2, seqlen=8):
    x = torch.randint(0, VOCAB, (bsz, seqlen))
    labels = torch.randint(0, VOCAB, (bsz, seqlen))
    return {"input_ids": x, "labels": labels, "attention_mask": torch.ones_like(x)}


def test_fp16_without_scaler_raises():
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = make_batch()
    with pytest.raises(ValueError, match="GradScaler"):
        run_training_step(m, opt, None, batch, VOCAB, 1.0, amp_dtype="fp16", scaler=None)


def test_bf16_with_scaler_raises():
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = make_batch()
    fake_scaler = object()  # no hace falta un GradScaler real para probar el guard de validacion
    with pytest.raises(ValueError, match="BF16"):
        run_training_step(m, opt, None, batch, VOCAB, 1.0, amp_dtype="bf16", scaler=fake_scaler)


def test_bf16_no_scaler_runs_structurally_on_cpu():
    """bf16 SI corre en CPU (autocast lo soporta) -- valida que la ruta
    bf16-sin-scaler ejecuta forward+backward+step sin error y produce una
    loss finita. Esto es una validacion estructural real, no un mock."""
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = make_batch()
    loss, logits = run_training_step(m, opt, None, batch, VOCAB, 1.0, amp_dtype="bf16", scaler=None)
    assert loss == loss and loss not in (float("inf"), float("-inf"))  # no NaN/Inf
    assert logits.shape == (2, 8, VOCAB)


def test_none_dtype_is_plain_fp32_step():
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = make_batch()
    loss, logits = run_training_step(m, opt, None, batch, VOCAB, 1.0, amp_dtype=None, scaler=None)
    assert loss == loss
    assert logits.dtype == torch.float32


def test_fp16_cuda_path_marked_pending_colab():
    """fp16+GradScaler real requiere CUDA -- no se puede validar en este
    entorno (torch build sin CUDA). Se documenta explicitamente en vez de
    fingir que se probo."""
    if torch.cuda.is_available():
        pytest.skip("Hay CUDA disponible -- si este skip nunca se activa en Colab, "
                     "reemplazar por una prueba real de fp16+GradScaler ahi.")
    pytest.skip("GPU_RUNTIME_VALIDATION_PENDING_COLAB: fp16+GradScaler no puede probarse sin CUDA real.")
