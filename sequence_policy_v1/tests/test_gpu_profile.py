"""Pruebas de gpu_profile.py: deteccion (CPU real -> CPU_LOCAL explicito,
nunca inventa T4/L4/A100), valores de los 3 perfiles de GPU documentados, y
el loop de fallback ante OOM (simulado -- no hay CUDA real en este entorno,
asi que se simula OutOfMemoryError directamente en vez de provocarlo)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.gpu_profile import GPU_PROFILES, OOMFallbackExhausted, detect_gpu_profile, run_with_oom_fallback


def test_cpu_local_never_claims_gpu_dtype():
    info = detect_gpu_profile()
    if torch.cuda.is_available():
        pytest.skip("Hay CUDA real disponible -- este test es especificamente para el caso sin GPU.")
    assert info["profile_name"] == "CPU_LOCAL"
    assert info["dtype"] == "fp32"
    assert info["use_scaler"] is False
    assert info["cuda_available"] is False
    assert info["gpu_name"] is None


def test_t4_profile_uses_fp16_and_scaler():
    p = GPU_PROFILES["T4"]
    assert p["dtype"] == "fp16"
    assert p["use_scaler"] is True


def test_l4_and_a100_use_bf16_no_scaler():
    for name in ("L4", "A100"):
        p = GPU_PROFILES[name]
        assert p["dtype"] == "bf16"
        assert p["use_scaler"] is False


def test_a100_has_largest_effective_batch():
    assert GPU_PROFILES["A100"]["effective_batch_tokens_target"] > GPU_PROFILES["L4"]["effective_batch_tokens_target"]
    assert GPU_PROFILES["L4"]["effective_batch_tokens_target"] > GPU_PROFILES["T4"]["effective_batch_tokens_target"]


def test_oom_fallback_reduces_microbatch_and_succeeds():
    """Simula 2 OOM consecutivos y despues exito -- verifica que el
    microbatch se reduce a la mitad y grad_accum se duplica cada vez, sin
    ocultar los intentos fallidos."""
    calls = []

    def fake_step(microbatch, grad_accum):
        calls.append((microbatch, grad_accum))
        if len(calls) < 3:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return "ok"

    result, info = run_with_oom_fallback(fake_step, initial_microbatch=32, min_microbatch=1)
    assert result == "ok"
    assert calls == [(32, 1), (16, 2), (8, 4)]
    assert info["microbatch"] == 8
    assert info["grad_accum_multiplier"] == 4
    assert len(info["attempts"]) == 2


def test_oom_fallback_raises_when_exhausted_at_min_microbatch():
    def always_oom(microbatch, grad_accum):
        raise torch.cuda.OutOfMemoryError("simulated persistent OOM")

    with pytest.raises(OOMFallbackExhausted):
        run_with_oom_fallback(always_oom, initial_microbatch=2, min_microbatch=1, max_retries=5)


def test_oom_fallback_does_not_retry_forever():
    calls = []

    def always_oom(microbatch, grad_accum):
        calls.append(microbatch)
        raise torch.cuda.OutOfMemoryError("simulated")

    with pytest.raises(OOMFallbackExhausted):
        run_with_oom_fallback(always_oom, initial_microbatch=1024, min_microbatch=1, max_retries=4)
    assert len(calls) <= 4
