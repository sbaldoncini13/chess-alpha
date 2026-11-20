"""Loop de entrenamiento generalizado (TINY local o BASE/LARGE en Colab).

Guard explicito: perfiles distintos de TINY se niegan a correr en CPU (mismo
patron que chess_alpha/37_train_v1_1.py con Epoch4 v1.1) -- evita que alguien
lance sin querer un entrenamiento BASE completo en la maquina local."""
import time

import torch
import torch.nn.functional as F

from sequence_policy.checkpointing import save_checkpoint
from sequence_policy.dataset_reader import SequenceWindowDataset


class TokenBudgetExceeded(Exception):
    pass


def assert_device_allowed(profile_name: str, device: str):
    if profile_name.upper() != "TINY" and device == "cpu":
        raise RuntimeError(
            f"Perfil '{profile_name}' no puede entrenarse en CPU local -- solo TINY esta "
            f"autorizado para smoke local. BASE/LARGE requieren GPU (Colab). Deteniendo."
        )


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine.item()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


_AMP_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None, None: None}


def _validate_amp_args(amp_dtype, scaler):
    if amp_dtype == "fp16" and scaler is None:
        raise ValueError("amp_dtype='fp16' requiere un GradScaler real (fp16 sin "
                          "loss scaling diverge). Si la GPU no soporta bf16 bien, usar fp16+scaler.")
    if amp_dtype == "bf16" and scaler is not None:
        raise ValueError("amp_dtype='bf16' no debe usarse con GradScaler (BF16 no necesita loss "
                          "scaling; usar un scaler ahi es un error de configuracion, no una opcion valida).")


def forward_backward_step(model, batch, vocab_size, grad_accum_steps=1, label_smoothing=0.0,
                           scaler=None, amp_dtype=None):
    """UN microbatch: forward + loss/grad_accum_steps + backward. NUNCA toca
    optimizer.step()/scheduler.step()/zero_grad() -- eso es responsabilidad de
    optimizer_step_and_reset(), llamada solo cuando se completa un grupo de
    acumulacion real. Devuelve la loss SIN dividir (para logging/promedios),
    aunque el backward si uso la version dividida.

    Bug corregido: antes run_training_step hacia optimizer.step()/scheduler.step()
    en CADA microbatch sin importar grad_accum_steps, haciendo que el batch
    efectivo real fuera solo el microbatch, no effective_batch_tokens_target
    (encontrado por smoke test real en T4: global_step avanzaba 1:1 con
    microbatches en vez de cada ~49)."""
    _validate_amp_args(amp_dtype, scaler)
    device_type = next(model.parameters()).device.type
    torch_dtype = _AMP_DTYPE_MAP[amp_dtype]

    if torch_dtype is not None:
        with torch.autocast(device_type=device_type, dtype=torch_dtype):
            logits, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch["labels"].view(-1),
                                    ignore_index=SequenceWindowDataset.IGNORE_INDEX,
                                    label_smoothing=label_smoothing)
        loss_to_backward = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()
    else:
        logits, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(logits.view(-1, vocab_size), batch["labels"].view(-1),
                                ignore_index=SequenceWindowDataset.IGNORE_INDEX,
                                label_smoothing=label_smoothing)
        (loss / grad_accum_steps).backward()
    return loss.item(), logits


def optimizer_step_and_reset(model, optimizer, scheduler, grad_clip_norm, scaler=None,
                              partial_group_rescale=None):
    """Se llama SOLO al completar un grupo de acumulacion (grad_accum_steps
    microbatches) o al final de la epoca con un grupo parcial. En ese ultimo
    caso, partial_group_rescale = grad_accum_steps/tamano_real_del_grupo
    renormaliza los gradientes acumulados (que se dividieron por
    grad_accum_steps microbatch a microbatch, no por el tamano real del
    grupo parcial, mas chico) antes de aplicar el clip+step."""
    if scaler is not None:
        scaler.unscale_(optimizer)
        if partial_group_rescale is not None:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(partial_group_rescale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if partial_group_rescale is not None:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(partial_group_rescale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def run_training_step(model, optimizer, scheduler, batch, vocab_size, grad_clip_norm,
                       label_smoothing=0.0, scaler=None, amp_dtype=None):
    """Wrapper de compatibilidad hacia atras -- UN microbatch = UN optimizer
    step completo (grad_accum_steps=1), exactamente el comportamiento
    original. Usado por scripts/run_train.py (piloto, YA ENTRENADO, no se
    toca ni se relanza) y por pruebas existentes. El full training NUEVO usa
    forward_backward_step + optimizer_step_and_reset directamente, nunca esta
    funcion, para poder acumular de verdad."""
    loss, logits = forward_backward_step(model, batch, vocab_size, grad_accum_steps=1,
                                          label_smoothing=label_smoothing, scaler=scaler, amp_dtype=amp_dtype)
    optimizer_step_and_reset(model, optimizer, scheduler, grad_clip_norm, scaler=scaler)
    return loss, logits


def autotune_microbatch(model, sample_batch_fn, vocab_size, device, max_microbatch=256,
                         vram_safety_margin_pct=15):
    """Duplica el microbatch hasta OOM o hasta max_microbatch, reservando el
    margen de VRAM pedido (seccion 18). En CPU devuelve un valor conservador
    fijo sin probar (no tiene sentido autotune de VRAM sin GPU)."""
    if device == "cpu":
        return {"microbatch": 8, "method": "cpu_fixed_no_autotune"}

    microbatch = 4
    while microbatch <= max_microbatch:
        try:
            batch = sample_batch_fn(microbatch)
            with torch.no_grad():
                model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_pct = 100 * free_mem / total_mem
            if free_pct < vram_safety_margin_pct:
                microbatch //= 2
                break
            microbatch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            microbatch //= 2
            break
    return {"microbatch": max(1, microbatch), "method": "gpu_probe"}
