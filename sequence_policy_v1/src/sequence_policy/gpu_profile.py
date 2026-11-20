"""Deteccion de GPU y politica de precision para el full training (Colab).

En CPU local esto SIEMPRE devuelve profile_name="CPU_LOCAL" -- nunca inventa
un perfil T4/L4/A100 sin GPU real detectada. La validacion real de estos
perfiles en hardware CUDA queda pendiente de Colab (ver GPU_RUNTIME_VALIDATION_PENDING_COLAB
en los reportes de build)."""
import torch

# batch efectivo objetivo en TOKENS reales (no windows) -- igual que los
# configs/train_full_{t4,l4,a100}.yaml de Etapa E, consolidado aca para que
# el codigo y los YAML no diverjan.
GPU_PROFILES = {
    "T4": {"vram_gb_min": 14, "vram_gb_max": 17, "dtype": "fp16", "use_scaler": True,
           "gradient_checkpointing": True, "effective_batch_tokens_target": 32768,
           "microbatch_base": 8, "vram_safety_margin_pct": 15},
    "L4": {"vram_gb_min": 20, "vram_gb_max": 25, "dtype": "bf16", "use_scaler": False,
           "gradient_checkpointing": False, "effective_batch_tokens_target": 49152,
           "microbatch_base": 16, "vram_safety_margin_pct": 15},
    "A100": {"vram_gb_min": 35, "vram_gb_max": 90, "dtype": "bf16", "use_scaler": False,
             "gradient_checkpointing": False, "effective_batch_tokens_target": 65536,
             "microbatch_base": 32, "vram_safety_margin_pct": 10},
}


def _match_profile_by_name(gpu_name: str):
    name_upper = gpu_name.upper()
    if "T4" in name_upper:
        return "T4"
    if "L4" in name_upper and "A100" not in name_upper:
        return "L4"
    if "A100" in name_upper:
        return "A100"
    return None


def detect_gpu_profile() -> dict:
    """No asume nada: si no hay CUDA, profile_name='CPU_LOCAL' y dtype='fp32'
    explicitamente -- nunca se afirma bf16/fp16 sin GPU real que lo soporte."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        info.update({
            "profile_name": "CPU_LOCAL", "gpu_name": None, "vram_gb": None,
            "cuda_version": None, "dtype": "fp32", "use_scaler": False,
            "gradient_checkpointing": False, "effective_batch_tokens_target": 4096,
            "microbatch_base": 4, "vram_safety_margin_pct": None,
            "note": "Sin GPU -- solo valido para smoke estructural TINY, nunca para BASE/LARGE real.",
        })
        return info

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(total_vram_bytes / (1024 ** 3), 1)
    cuda_version = torch.version.cuda

    matched = _match_profile_by_name(gpu_name)
    if matched is None:
        # GPU CUDA real pero no reconocida por nombre -- perfil conservador
        # generico, NUNCA se afirma bf16 "porque si": se decide por capacidad
        # real (Ampere+ = arquitectura >= 8.0 soporta bf16 con buen rendimiento).
        major, _minor = torch.cuda.get_device_capability(0)
        bf16_capable = major >= 8
        info.update({
            "profile_name": "OTHER_CUDA", "gpu_name": gpu_name, "vram_gb": vram_gb,
            "cuda_version": cuda_version,
            "dtype": "bf16" if bf16_capable else "fp16",
            "use_scaler": not bf16_capable,
            "gradient_checkpointing": vram_gb < 20,
            "effective_batch_tokens_target": 32768 if vram_gb < 20 else 49152,
            "microbatch_base": 8 if vram_gb < 20 else 16,
            "vram_safety_margin_pct": 15,
            "note": f"GPU no reconocida por nombre ('{gpu_name}'), perfil conservador por "
                    f"compute capability {major}.{_minor} (bf16_capable={bf16_capable}).",
        })
        return info

    profile = GPU_PROFILES[matched]
    info.update({
        "profile_name": matched, "gpu_name": gpu_name, "vram_gb": vram_gb,
        "cuda_version": cuda_version, **profile,
        "note": f"Perfil {matched} reconocido por nombre de GPU.",
    })
    return info


class OOMFallbackExhausted(Exception):
    pass


def run_with_oom_fallback(train_step_fn, initial_microbatch: int, min_microbatch: int = 1,
                           max_retries: int = 6, on_event=None):
    """Ejecuta train_step_fn(microbatch, grad_accum_multiplier) reduciendo el
    microbatch a la mitad y duplicando el multiplicador de gradient
    accumulation (para conservar el batch efectivo) cada vez que aparece un
    OutOfMemoryError real de CUDA. No reintenta indefinidamente -- corta en
    max_retries y no oculta el error final."""
    microbatch = initial_microbatch
    grad_accum_multiplier = 1
    attempts = []
    for attempt in range(max_retries):
        try:
            result = train_step_fn(microbatch, grad_accum_multiplier)
            if attempts:
                # solo se emite evento cuando hubo un OOM real y se recupero -- el caso
                # normal (attempt==0, sin OOM) NO llama on_event en cada microbatch
                # (antes lo hacia, generaba un evento por batch sin ningun valor real)
                if on_event:
                    on_event({"event": "oom_fallback_recovered", "attempt": attempt,
                              "final_microbatch": microbatch, "grad_accum_multiplier": grad_accum_multiplier,
                              "attempts": attempts})
                event_label = "oom_fallback_recovered"
            else:
                event_label = "gpu_probe_success"
            return result, {"microbatch": microbatch, "grad_accum_multiplier": grad_accum_multiplier,
                             "attempts": attempts, "event_label": event_label}
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            diagnostic = {"attempt": attempt, "microbatch_tried": microbatch, "error": str(exc)}
            attempts.append(diagnostic)
            if on_event:
                on_event({"event": "oom_detected", **diagnostic})
            if microbatch <= min_microbatch:
                raise OOMFallbackExhausted(
                    f"OOM persistente incluso en microbatch minimo ({min_microbatch}) tras "
                    f"{attempt + 1} intentos. No se oculta el error, se detiene. attempts={attempts}"
                ) from exc
            microbatch = max(min_microbatch, microbatch // 2)
            grad_accum_multiplier *= 2
    raise OOMFallbackExhausted(f"Se agotaron {max_retries} reintentos de OOM fallback. attempts={attempts}")
