"""Calculo del presupuesto de pasos/tokens para el full training y tabla de
LR -- reutiliza la MISMA formula de build_scheduler (trainer.py) para que el
reporte y el scheduler real nunca diverjan.

Definiciones exactas (no supuestas, calculadas desde windows_train.parquet /
windows_manifest.json):
  - EPOCH_TARGETS = 73.977.508  (targets/labels de loss, una epoca de train,
    esta es la cifra que el usuario fijo como definicion de "una epoca")
  - EPOCH_INPUT_TOKENS = 74.877.723  (suma de window_length -- tokens de
    ENTRADA reales de una pasada completa; ligeramente mayor que EPOCH_TARGETS
    porque cada ventana que arranca en context_start=0 incluye el token BOS,
    que es entrada pero nunca un target. Diferencia esperada ~1.2%, no es un
    error.)

tokens_seen (tanto en el piloto como en el full training) se mide en TOKENS
DE ENTRADA (attention_mask.sum()), igual que run_train.py -- por eso el
presupuesto de pasos usa EPOCH_INPUT_TOKENS, no EPOCH_TARGETS."""
import math

import torch

EPOCH_TARGETS = 73_977_508
EPOCH_INPUT_TOKENS = 74_877_723
PILOT_TOKENS_SEEN = 5_000_327
PILOT_GLOBAL_STEP = 7532


def compute_total_steps(effective_batch_tokens_target: int, epoch_input_tokens: int = EPOCH_INPUT_TOKENS) -> int:
    return math.ceil(epoch_input_tokens / effective_batch_tokens_target)


def lr_at_step(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """Identica a la lambda interna de trainer.build_scheduler -- se llama
    ese mismo scheduler para construir esta tabla, nunca se reimplementa la
    formula por separado (evita que reporte y codigo real diverjan)."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy_param], lr=base_lr)
    from sequence_policy.trainer import build_scheduler
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)
    for _ in range(step):
        optimizer.step()
        scheduler.step()
    return optimizer.param_groups[0]["lr"]


def generate_lr_table(base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> list:
    """Tabla en start/warmup_end/25%/50%/75%/final -- construida avanzando UN
    scheduler real paso a paso (no formula cerrada aparte) para garantizar
    que coincide exactamente con lo que el entrenamiento real hara."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([dummy_param], lr=base_lr)
    from sequence_policy.trainer import build_scheduler
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)

    checkpoints = {
        "start (step=0)": 0,
        f"warmup_end (step={warmup_steps})": warmup_steps,
        f"25% (step={round(total_steps*0.25)})": round(total_steps * 0.25),
        f"50% (step={round(total_steps*0.50)})": round(total_steps * 0.50),
        f"75% (step={round(total_steps*0.75)})": round(total_steps * 0.75),
        f"final (step={total_steps})": total_steps,
    }
    sorted_checkpoints = sorted(checkpoints.items(), key=lambda kv: kv[1])

    rows = []
    current_step = 0
    lr_now = optimizer.param_groups[0]["lr"]
    for label, target_step in sorted_checkpoints:
        while current_step < target_step:
            optimizer.step()
            scheduler.step()
            current_step += 1
        lr_now = optimizer.param_groups[0]["lr"]
        assert lr_now == lr_now, f"LR es NaN en {label}"
        assert lr_now >= 0, f"LR es negativo en {label}: {lr_now}"
        rows.append({"checkpoint": label, "step": target_step, "lr": round(lr_now, 8)})
    return rows


def compute_budget_report(gpu_profiles: dict, warmup_pct: float = 0.03, min_lr_ratio: float = 0.1,
                           base_lr: float = 1e-4) -> dict:
    report = {
        "epoch_targets": EPOCH_TARGETS,
        "epoch_input_tokens": EPOCH_INPUT_TOKENS,
        "pilot_tokens_seen": PILOT_TOKENS_SEEN,
        "pilot_global_step": PILOT_GLOBAL_STEP,
        "approx_cumulative_tokens_including_pilot": PILOT_TOKENS_SEEN + EPOCH_TARGETS,
        "base_lr": base_lr,
        "warmup_pct": warmup_pct,
        "min_lr_ratio": min_lr_ratio,
        "profiles": {},
    }
    for profile_name, profile in gpu_profiles.items():
        eff_tokens = profile["effective_batch_tokens_target"]
        total_steps = compute_total_steps(eff_tokens)
        warmup_steps = max(1, round(total_steps * warmup_pct))
        lr_table = generate_lr_table(base_lr, warmup_steps, total_steps, min_lr_ratio)
        report["profiles"][profile_name] = {
            "effective_batch_tokens_target": eff_tokens,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "lr_table": lr_table,
        }
    return report
