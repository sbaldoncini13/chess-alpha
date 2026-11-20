"""Losses configurables del entrenamiento v1.1 (secciones 8-11 del spec).

PolicyLoss soporta target duro (indice unico + label smoothing confinado a
las jugadas LEGALES de esa posicion, nunca a todo el vocabulario) y target
blando (distribucion ya normalizada sobre varias jugadas) en el mismo
batch, seleccionando por fila segun `is_soft`. En ambos casos aplica la
mascara legal a los logits antes de softmax, para que la red nunca
desperdicie masa en jugadas ilegales o fuera de vocabulario.

ValueLossModule soporta MSE o SmoothL1 (Huber) configurable.

TotalLoss combina ambas con pesos configurables y expone las magnitudes
individuales + normas de gradiente por modulo para diagnostico (seccion 11:
"no decidir unicamente por la magnitud numerica de las losses")."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing debe estar en [0,1), dio {label_smoothing}")
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, legal_mask: torch.Tensor, hard_target: torch.Tensor,
                soft_target: torch.Tensor, is_soft: torch.Tensor) -> torch.Tensor:
        """logits: (B,V). legal_mask: (B,V) bool. hard_target: (B,) long.
        soft_target: (B,V) float, ya normalizado por fila (0 en filas hard).
        is_soft: (B,) bool, que filas usan target blando."""
        # torch.finfo(dtype).min en vez de un -1e9 fijo: bajo autocast/AMP
        # `logits` llega en float16 (rango +-65504), y -1e9 desborda ahi
        # (RuntimeError al convertir). float16 "min" ya es lo bastante
        # negativo para que exp() colapse a 0 en el softmax de abajo.
        masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        log_probs = F.log_softmax(masked_logits, dim=1)

        B, V = logits.shape
        device = logits.device
        per_row_loss = torch.zeros(B, device=device)

        hard_rows = (~is_soft)
        if hard_rows.any():
            legal_count = legal_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            # smoothing distribuido solo sobre jugadas legales (nunca sobre
            # todo el vocabulario: log_prob de una jugada ilegal es ~-1e9 y
            # ponerle masa ahi rompe la loss numericamente)
            smooth_val = self.label_smoothing / (legal_count - 1).clamp(min=1)
            target_dist = legal_mask.float() * smooth_val
            target_dist.scatter_(1, hard_target.unsqueeze(1), 1.0 - self.label_smoothing)
            hard_loss = -(target_dist * log_probs).sum(dim=1)
            per_row_loss = torch.where(hard_rows, hard_loss, per_row_loss)

        soft_rows = is_soft
        if soft_rows.any():
            # renormalizar por las dudas (candidatas fuera de la mascara legal
            # no deberian existir, pero no confiar ciegamente en los datos)
            masked_soft = soft_target * legal_mask.float()
            row_sums = masked_soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
            masked_soft = masked_soft / row_sums
            soft_loss = -(masked_soft * log_probs).sum(dim=1)
            per_row_loss = torch.where(soft_rows, soft_loss, per_row_loss)

        return per_row_loss.mean()


class ValueLoss(nn.Module):
    SUPPORTED = ("mse", "smooth_l1")

    def __init__(self, loss_type: str = "smooth_l1", smooth_l1_beta: float = 0.1):
        super().__init__()
        if loss_type not in self.SUPPORTED:
            raise ValueError(f"value loss_type '{loss_type}' no soportado, opciones: {self.SUPPORTED}")
        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta

    def forward(self, value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(value_pred, value_target)
        return F.smooth_l1_loss(value_pred, value_target, beta=self.smooth_l1_beta)


class TotalLoss(nn.Module):
    """Combina policy + value con pesos configurables. `forward` devuelve
    (loss_total, diagnostics) donde diagnostics trae las magnitudes
    individuales (sin el peso aplicado) para poder inspeccionar el balance
    entre cabezas sin tener que adivinar a partir de la loss combinada."""

    def __init__(self, policy_loss_fn: PolicyLoss, value_loss_fn: ValueLoss,
                 policy_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        if policy_weight < 0 or value_weight < 0:
            raise ValueError(f"policy_weight/value_weight deben ser >= 0, dio {policy_weight}, {value_weight}")
        self.policy_loss_fn = policy_loss_fn
        self.value_loss_fn = value_loss_fn
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(self, policy_logits, legal_mask, hard_target, soft_target, is_soft, value_pred, value_target):
        policy_loss = self.policy_loss_fn(policy_logits, legal_mask, hard_target, soft_target, is_soft)
        value_loss = self.value_loss_fn(value_pred, value_target)
        total = self.policy_weight * policy_loss + self.value_weight * value_loss
        diagnostics = {
            "policy_loss": policy_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "policy_loss_weighted": (self.policy_weight * policy_loss).detach().item(),
            "value_loss_weighted": (self.value_weight * value_loss).detach().item(),
        }
        return total, diagnostics


def grad_norms_by_module(model) -> dict:
    """Norma L2 del gradiente acumulado en stem/conv_blocks(trunk residual)/
    transformer_blocks(trunk atencion)/policy_head/value_head. Se llama
    DESPUES de loss.backward() y ANTES de optimizer.step()/zero_grad().
    Sirve para decidir si value_weight es demasiado chico (seccion 11:
    "si value tiene una contribucion demasiado pequena, ajustar el peso")
    con evidencia en vez de solo mirar la magnitud de la loss."""
    groups = {
        "stem": model.stem,
        "conv_blocks": model.conv_blocks,
        "transformer_blocks": model.transformer_blocks,
        "policy_head": model.policy_head,
        "value_head": model.value_head,
    }
    norms = {}
    for name, module in groups.items():
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().norm(2).item() ** 2
        norms[name] = total_sq ** 0.5
    return norms
