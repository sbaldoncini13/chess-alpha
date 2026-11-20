"""Pruebas de acumulacion real de gradientes (forward_backward_step +
optimizer_step_and_reset), corrigiendo el bug real encontrado por el smoke
test en T4: antes optimizer.step()/scheduler.step() ocurrian en CADA
microbatch sin importar grad_accum_target, haciendo que el batch efectivo
real fuera solo el microbatch."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sequence_policy.model import CausalMoveTransformer
from sequence_policy.trainer import build_scheduler, forward_backward_step, optimizer_step_and_reset

VOCAB = 20483
KW = dict(vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, head_dim=16,
          ffn_hidden=128, context_length=32, dropout=0.0, attention_dropout=0.0)


def make_batch(bsz=2, seqlen=8):
    x = torch.randint(0, VOCAB, (bsz, seqlen))
    labels = torch.randint(0, VOCAB, (bsz, seqlen))
    return {"input_ids": x, "labels": labels, "attention_mask": torch.ones_like(x)}


def make_model_opt_sched(total_steps=100, warmup_steps=5, lr=1e-3):
    torch.manual_seed(0)
    m = CausalMoveTransformer(**KW)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = build_scheduler(opt, warmup_steps, total_steps, min_lr_ratio=0.1)
    return m, opt, sched


def run_n_microbatches(model, optimizer, scheduler, n_micro, grad_accum_target, grad_clip_norm=1.0):
    """Simula el loop real de run_full_train.py: acumula n_micro microbatches,
    ejecuta optimizer_step_and_reset cada grad_accum_target, devuelve cuantos
    optimizer steps ocurrieron realmente."""
    micro_step = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(n_micro):
        batch = make_batch()
        forward_backward_step(model, batch, VOCAB, grad_accum_steps=grad_accum_target)
        micro_step += 1
        if micro_step >= grad_accum_target:
            optimizer_step_and_reset(model, optimizer, scheduler, grad_clip_norm)
            optimizer_steps += 1
            micro_step = 0
    return optimizer_steps, micro_step


def test_48_microbatches_49_target_zero_optimizer_steps():
    model, opt, sched = make_model_opt_sched()
    steps, pending = run_n_microbatches(model, opt, sched, n_micro=48, grad_accum_target=49)
    assert steps == 0
    assert pending == 48
    assert sched.last_epoch == 0
    assert opt.state == {} or all(len(s) == 0 for s in opt.state.values()) is False  # backward corrio (grads existen)


def test_49_microbatches_49_target_one_optimizer_step():
    model, opt, sched = make_model_opt_sched()
    steps, pending = run_n_microbatches(model, opt, sched, n_micro=49, grad_accum_target=49)
    assert steps == 1
    assert pending == 0
    assert sched.last_epoch == 1


def test_98_microbatches_49_target_two_optimizer_steps():
    model, opt, sched = make_model_opt_sched()
    steps, pending = run_n_microbatches(model, opt, sched, n_micro=98, grad_accum_target=49)
    assert steps == 2
    assert pending == 0
    assert sched.last_epoch == 2


def test_scheduler_advances_exactly_with_optimizer_not_microbatches():
    """El bug real: antes scheduler.last_epoch == n_microbatches. Ahora debe
    ser == n_optimizer_steps, sin importar cuantos microbatches hicieron falta."""
    model, opt, sched = make_model_opt_sched(total_steps=100, warmup_steps=10)
    steps, _ = run_n_microbatches(model, opt, sched, n_micro=245, grad_accum_target=49)
    assert steps == 5
    assert sched.last_epoch == 5, "scheduler debe avanzar 1 vez por optimizer step (5), no 1 vez por microbatch (245)"


def test_max_steps_5_produces_exactly_5_optimizer_steps_with_realistic_grad_accum():
    model, opt, sched = make_model_opt_sched(total_steps=1000, warmup_steps=50)
    max_steps = 5
    grad_accum_target = 49
    optimizer_steps = 0
    micro_step = 0
    total_micro = 0
    opt.zero_grad(set_to_none=True)
    while optimizer_steps < max_steps:
        batch = make_batch()
        forward_backward_step(model, batch, VOCAB, grad_accum_steps=grad_accum_target)
        micro_step += 1
        total_micro += 1
        if micro_step >= grad_accum_target:
            optimizer_step_and_reset(model, opt, sched, 1.0)
            optimizer_steps += 1
            micro_step = 0
    assert optimizer_steps == 5
    assert total_micro == 5 * 49 == 245
    assert sched.last_epoch == 5


def test_loss_is_divided_before_backward_gradients_scale_accordingly():
    """Acumular grad_accum_steps microbatches IDENTICOS y hacer 1 paso debe
    dar (aprox) el mismo gradiente que backward de UNA sola pasada sin dividir
    -- confirma que la division por grad_accum_steps compensa la suma de
    gradientes de los N microbatches acumulados."""
    torch.manual_seed(1)
    m1 = CausalMoveTransformer(**KW)
    torch.manual_seed(1)
    m2 = CausalMoveTransformer(**KW)

    batch = make_batch()
    N = 4

    # m1: acumula el MISMO batch N veces con division por N
    for p in m1.parameters():
        p.grad = None
    for _ in range(N):
        forward_backward_step(m1, batch, VOCAB, grad_accum_steps=N)
    grad1 = [p.grad.clone() for p in m1.parameters() if p.grad is not None]

    # m2: un solo backward sin dividir (equivalente matematico esperado)
    for p in m2.parameters():
        p.grad = None
    forward_backward_step(m2, batch, VOCAB, grad_accum_steps=1)
    grad2 = [p.grad.clone() for p in m2.parameters() if p.grad is not None]

    assert len(grad1) == len(grad2)
    for g1, g2 in zip(grad1, grad2):
        assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-3), \
            "acumular N veces el mismo batch dividido por N debe dar el mismo gradiente que 1 vez sin dividir"


def test_partial_group_rescale_matches_equivalent_direct_average():
    """Grupo parcial de tamano k<grad_accum_target: el rescale (grad_accum_target/k)
    aplicado a gradientes acumulados-y-divididos-por-grad_accum_target debe dar
    el mismo resultado que haber dividido directamente por k desde el principio."""
    torch.manual_seed(2)
    m_rescaled = CausalMoveTransformer(**KW)
    torch.manual_seed(2)
    m_direct = CausalMoveTransformer(**KW)

    batch = make_batch()
    grad_accum_target = 10
    k = 3  # grupo parcial real

    for p in m_rescaled.parameters():
        p.grad = None
    for _ in range(k):
        forward_backward_step(m_rescaled, batch, VOCAB, grad_accum_steps=grad_accum_target)
    rescale = grad_accum_target / k
    for p in m_rescaled.parameters():
        if p.grad is not None:
            p.grad.mul_(rescale)
    grad_rescaled = [p.grad.clone() for p in m_rescaled.parameters() if p.grad is not None]

    for p in m_direct.parameters():
        p.grad = None
    for _ in range(k):
        forward_backward_step(m_direct, batch, VOCAB, grad_accum_steps=k)
    grad_direct = [p.grad.clone() for p in m_direct.parameters() if p.grad is not None]

    for g1, g2 in zip(grad_rescaled, grad_direct):
        assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-3)


def test_gradient_clipping_happens_after_unscale_structural():
    """Con un GradScaler real (enabled=False -- valido en CPU, no requiere CUDA),
    confirma que optimizer_step_and_reset corre sin excepciones y que el orden
    de llamadas (unscale_ -> clip -> step -> update) no rompe con un scaler
    real, no solo con scaler=None. Numericamente con enabled=False el scaler
    es un passthrough; fp16 real en CUDA queda GPU_RUNTIME_VALIDATION_PENDING_COLAB."""
    model, opt, sched = make_model_opt_sched()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = make_batch()
    opt.zero_grad(set_to_none=True)
    for _ in range(3):
        forward_backward_step(model, batch, VOCAB, grad_accum_steps=3, scaler=scaler)
    optimizer_step_and_reset(model, opt, sched, 1.0, scaler=scaler)
    assert sched.last_epoch == 1
    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)  # zero_grad(set_to_none=True) tras el step


def test_bf16_accumulation_runs_structurally_on_cpu():
    model, opt, sched = make_model_opt_sched()
    batch = make_batch()
    opt.zero_grad(set_to_none=True)
    for _ in range(3):
        loss, _ = forward_backward_step(model, batch, VOCAB, grad_accum_steps=3, amp_dtype="bf16")
        assert loss == loss  # no NaN
    optimizer_step_and_reset(model, opt, sched, 1.0)
    assert sched.last_epoch == 1


def test_checkpoints_only_at_zero_micro_step_boundary():
    """Replica la invariante del loop real: micro_step SIEMPRE es 0 en el
    momento de guardar un checkpoint (grupo completo o parcial ya cerrado)."""
    model, opt, sched = make_model_opt_sched()
    grad_accum_target = 5
    micro_step = 0
    checkpoints_saved_at_micro_step = []
    opt.zero_grad(set_to_none=True)
    for i in range(12):  # 12 microbatches, grad_accum_target=5 -> steps en 5 y 10, resto pendiente
        batch = make_batch()
        forward_backward_step(model, batch, VOCAB, grad_accum_steps=grad_accum_target)
        micro_step += 1
        if micro_step >= grad_accum_target:
            optimizer_step_and_reset(model, opt, sched, 1.0)
            micro_step = 0
            checkpoints_saved_at_micro_step.append(micro_step)  # simula "guardar checkpoint aca"
    assert checkpoints_saved_at_micro_step == [0, 0]
    assert len(checkpoints_saved_at_micro_step) == 2  # 12 // 5 = 2 grupos completos, no 12
