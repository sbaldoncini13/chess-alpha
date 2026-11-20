"""Loop de entrenamiento/evaluacion compartido entre el script definitivo
(37_train_v1_1.py) y el runner de ablations (38_run_ablations.py), para no
duplicar la logica de mezcla de batches / forward / backward / metricas."""

import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from chess_alpha.train_dataset import MixSpec, build_epoch_indices, MixedChessDataset, collate_batch
from chess_alpha.train_losses import PolicyLoss, ValueLoss, TotalLoss, grad_norms_by_module
from chess_alpha.train_optim import build_optimizer, build_scheduler, EMA
from chess_alpha.train_balance_sampler import build_balanced_general_indices


def build_train_rows(pools: dict, cfg, epoch_seed: int, enriched_general_pool: pd.DataFrame = None):
    """enriched_general_pool: solo requerido si cfg.data_mix.general_sampler
    == 'balanced' (ver chess_alpha.train_balance_sampler). Si es 'natural'
    (default), el bucket general se muestrea como siempre: permutacion
    simple del pool de train, sin reponderar por resultado/color."""
    n_general = len(enriched_general_pool) if cfg.data_mix.general_sampler == "balanced" else len(pools["general_train"])
    mix = MixSpec(n_general=n_general, data_mix_cfg=cfg.data_mix)
    indices, mix_report = build_epoch_indices(
        mix,
        n_corrective_policy_pool=len(pools["corrective_policy_train"]),
        n_corrective_value_pool=len(pools["corrective_value_train"]),
        n_tactical_pool=len(pools["tactical_train"]),
        oversampling_max=cfg.data_mix.oversampling_max,
        seed=epoch_seed,
    )

    if cfg.data_mix.general_sampler == "balanced":
        if enriched_general_pool is None:
            raise ValueError("general_sampler='balanced' pero no se paso enriched_general_pool.")
        balanced_idx, balance_report = build_balanced_general_indices(
            enriched_general_pool, n_target=mix.n_general, seed=epoch_seed,
            max_positions_per_game=cfg.data_mix.balance_max_positions_per_game,
            oversampling_max=cfg.data_mix.balance_oversampling_max,
        )
        general_rows = enriched_general_pool.iloc[balanced_idx]
        mix_report["balance_sampler_report"] = balance_report
    else:
        general_rows = pools["general_train"].iloc[indices["general"]]

    parts = [
        general_rows,
        pools["corrective_policy_train"].iloc[indices["corrective_policy"]],
        pools["corrective_value_train"].iloc[indices["corrective_value"]],
        pools["tactical_train"].iloc[indices["corrective_critical"]],
        pools["tactical_train"].iloc[indices["tactical"]],
    ]
    rows = pd.concat(parts, ignore_index=True)
    return rows, mix_report


def build_val_rows(pools: dict) -> pd.DataFrame:
    return pd.concat(
        [pools["general_val"], pools["corrective_val"], pools["tactical_val"]],
        ignore_index=True,
    )


def build_loss_fn(cfg):
    policy_loss_fn = PolicyLoss(label_smoothing=cfg.policy_target.label_smoothing)
    value_loss_fn = ValueLoss(loss_type=cfg.value_target.loss_type, smooth_l1_beta=cfg.value_target.smooth_l1_beta)
    return TotalLoss(policy_loss_fn, value_loss_fn, cfg.loss.policy_weight, cfg.loss.value_weight)


def topk_hit(logits, hard_target, soft_target, is_soft, k):
    """Contra el target de mayor peso de cada fila (hard target o el
    argmax del target blando)."""
    reference = torch.where(is_soft, soft_target.argmax(dim=1), hard_target)
    topk_pred = logits.topk(k, dim=1).indices
    return (topk_pred == reference.unsqueeze(1)).any(dim=1).float().mean().item()


def run_epoch(model, loader, total_loss_fn, device, optimizer=None, scheduler=None,
              grad_clip_norm=1.0, train=True, max_steps=None, ema=None, use_amp=False,
              grad_accum_steps=1, log_every=None, checkpoint_every_seconds=None, checkpoint_fn=None,
              scaler=None):
    """checkpoint_fn(step): callback opcional invocado cada
    `checkpoint_every_seconds` durante train=True, para poder reanudar a
    mitad de epoch si Colab se desconecta (seccion colab.* del spec:
    'proteccion ante desconexiones'). No reemplaza el checkpoint de fin de
    epoch, es un salvavidas intermedio.

    scaler: GradScaler externo y persistente entre epochs (para poder
    guardar/restaurar su estado en el checkpoint). Si no se pasa, se crea
    uno nuevo por llamada (comportamiento anterior, valido para val o para
    scripts que no necesitan resumir)."""
    model.train(train)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0, "top3": 0.0, "n": 0}
    grad_norm_samples = []
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=use_amp)
    t0 = time.time()
    last_checkpoint = time.time()

    if train:
        optimizer.zero_grad()

    for step, (xb, mask, hard_t, soft_t, is_soft, value_t) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        xb, mask, hard_t = xb.to(device), mask.to(device), hard_t.to(device)
        soft_t, is_soft, value_t = soft_t.to(device), is_soft.to(device), value_t.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                policy_logits, value_pred = model(xb)
                loss, diag = total_loss_fn(policy_logits, mask, hard_t, soft_t, is_soft, value_pred, value_t)
                loss_to_backward = loss / grad_accum_steps

            if train:
                if use_amp:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                if (step + 1) % grad_accum_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    if len(grad_norm_samples) < 5:
                        grad_norm_samples.append(grad_norms_by_module(model))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    if ema is not None:
                        ema.update(model)

        bs = xb.size(0)
        totals["policy_loss"] += diag["policy_loss"] * bs
        totals["value_loss"] += diag["value_loss"] * bs
        with torch.no_grad():
            totals["top1"] += topk_hit(policy_logits, hard_t, soft_t, is_soft, 1) * bs
            totals["top3"] += topk_hit(policy_logits, hard_t, soft_t, is_soft, 3) * bs
        totals["n"] += bs

        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}: policy_loss={totals['policy_loss']/totals['n']:.4f} "
                  f"value_loss={totals['value_loss']/totals['n']:.4f} top1={totals['top1']/totals['n']:.4f}")

        if train and checkpoint_fn is not None and checkpoint_every_seconds and (time.time() - last_checkpoint > checkpoint_every_seconds):
            checkpoint_fn(step + 1)
            last_checkpoint = time.time()

    n = max(1, totals["n"])
    metrics = {
        "policy_loss": totals["policy_loss"] / n,
        "value_loss": totals["value_loss"] / n,
        "top1": totals["top1"] / n,
        "top3": totals["top3"] / n,
        "n_positions": totals["n"],
        "elapsed_seconds": time.time() - t0,
        "grad_norms_sample": grad_norm_samples,
    }
    return metrics


def make_loader(rows, vocab, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
    ds = MixedChessDataset(rows, vocab)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
