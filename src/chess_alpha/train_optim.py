"""Optimizer, scheduler y EMA para el entrenamiento v1.1 (secciones 12-14
del spec). LR distinto por grupo de capas (stem/trunk/heads), warmup +
cosine decay, y una copia EMA de los pesos opcional."""

import copy
import math

import torch


def build_optimizer(model, optim_cfg):
    """Grupos de parametros con LR propio: stem, trunk (conv_blocks +
    transformer_blocks + pos_embedding + transformer_norm), policy_head,
    value_head. weight_decay no se aplica a bias/BatchNorm/LayerNorm."""
    if optim_cfg.name != "adamw":
        raise ValueError(f"optimizer '{optim_cfg.name}' no soportado")

    def split_decay(module):
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "bn" in name.lower() or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    trunk_modules = [model.conv_blocks, model.transformer_blocks, model.transformer_norm]
    trunk_extra_params = [model.pos_embedding]

    param_groups = []
    for module, lr, label in (
        (model.stem, optim_cfg.lr_stem, "stem"),
        (model.policy_head, optim_cfg.lr_heads, "policy_head"),
        (model.value_head, optim_cfg.lr_heads, "value_head"),
    ):
        decay, no_decay = split_decay(module)
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": optim_cfg.weight_decay, "name": f"{label}_decay"})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{label}_no_decay"})

    trunk_decay, trunk_no_decay = [], []
    for module in trunk_modules:
        d, nd = split_decay(module)
        trunk_decay += d
        trunk_no_decay += nd
    trunk_decay += trunk_extra_params  # pos_embedding: parametro directo, con decay (no es bias/norm)
    param_groups.append({"params": trunk_decay, "lr": optim_cfg.lr_trunk, "weight_decay": optim_cfg.weight_decay, "name": "trunk_decay"})
    if trunk_no_decay:
        param_groups.append({"params": trunk_no_decay, "lr": optim_cfg.lr_trunk, "weight_decay": 0.0, "name": "trunk_no_decay"})

    return torch.optim.AdamW(param_groups)


class WarmupCosineScheduler:
    """LR multiplicativo: warmup lineal 0->1 durante `warmup_steps`, despues
    coseno de 1 -> (lr_min/lr_base) hasta `total_steps`. Un scheduler manual
    (no torch.optim.lr_scheduler) porque cada param_group tiene su propio
    lr_base y el warmup/coseno deben aplicar el mismo factor relativo a
    todos por igual."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, lr_min_ratio: float):
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.lr_min_ratio = lr_min_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0
        self._apply()

    def _factor(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.step_num / max(1, self.warmup_steps)
        progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine

    def _apply(self):
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self):
        self.step_num += 1
        self._apply()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
            "lr_min_ratio": self.lr_min_ratio, "base_lrs": self.base_lrs, "step_num": self.step_num,
        }

    def load_state_dict(self, state):
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.lr_min_ratio = state["lr_min_ratio"]
        self.base_lrs = state["base_lrs"]
        self.step_num = state["step_num"]
        self._apply()


def build_scheduler(optimizer, scheduler_cfg, total_steps: int):
    if scheduler_cfg.name == "none":
        return None
    if scheduler_cfg.name != "warmup_cosine":
        raise ValueError(f"scheduler '{scheduler_cfg.name}' no soportado")
    warmup_steps = int(total_steps * scheduler_cfg.warmup_pct)
    base_lr = max(g["lr"] for g in optimizer.param_groups) or 1.0
    lr_min_ratio = scheduler_cfg.lr_min / base_lr if base_lr > 0 else 0.0
    return WarmupCosineScheduler(optimizer, warmup_steps, total_steps, lr_min_ratio)


class EMA:
    """Copia EMA de los pesos del modelo. `update` se llama despues de cada
    optimizer.step(); `apply_to` / `restore` permiten evaluar con los pesos
    EMA sin perder los pesos de entrenamiento en curso."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def apply_to(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        if hasattr(self, "_backup"):
            model.load_state_dict(self._backup)
            del self._backup
