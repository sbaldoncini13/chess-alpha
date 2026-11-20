"""Configuracion central del entrenamiento v1.1 (checkpoint hibrido).

Toda la parametrizacion del pipeline (rutas, mezcla de datos, targets,
losses, optimizer, scheduler, estabilidad, runtime) vive en un YAML
(ver configs/config_v11.yaml y configs/config_v11_colab.yaml) y se carga
con `load_config`. `validate_config` corre antes de tocar cualquier dato o
modelo y aborta con un mensaje claro ante cualquiera de las condiciones de
parada pedidas: rutas faltantes, porcentajes que no suman, checkpoint o
vocabulario incompatibles, tensor con forma distinta a 23x8x8, pesos de
loss invalidos, opcion no soportada, poco espacio en disco, o GPU pedida
pero no disponible.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import torch
import yaml

SUPPORTED_OPTIMIZERS = ("adamw",)
SUPPORTED_SCHEDULERS = ("warmup_cosine", "none")
SUPPORTED_VALUE_LOSSES = ("mse", "smooth_l1")
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")
SUPPORTED_MODES = ("local", "colab")
EXPECTED_TENSOR_SHAPE = (23, 8, 8)


class ConfigError(ValueError):
    """Error de configuracion o de compatibilidad detectado antes de entrenar.
    Su mensaje debe ser suficiente para que el usuario entienda que corregir,
    sin tener que leer el traceback."""


@dataclasses.dataclass
class PathsConfig:
    checkpoint_in: str
    vocab_path: str
    dataset_general: str
    dataset_corrective: str
    dataset_tactical: str
    output_dir: str
    log_dir: str
    validation_set_path: str
    unified_pool_path: str


@dataclasses.dataclass
class ArchConfig:
    channels: int = 256
    num_conv_blocks: int = 6
    num_transformer_blocks: int = 4
    num_heads: int = 8


@dataclasses.dataclass
class DataMixConfig:
    pct_general: float = 0.75
    pct_corrective: float = 0.15
    pct_tactical: float = 0.10
    corrective_pct_policy: float = 0.50
    corrective_pct_value: float = 0.30
    corrective_pct_critical: float = 0.20
    oversampling_max: int = 5
    val_fraction: float = 0.10
    seed: int = 42
    general_sampler: str = "natural"          # natural | balanced (Punto 2/5: balanceo por resultado/color)
    balance_max_positions_per_game: int = 8
    balance_oversampling_max: int = 4


@dataclasses.dataclass
class PolicyTargetConfig:
    label_smoothing: float = 0.05
    soft_temperature_cp: float = 100.0


@dataclasses.dataclass
class ValueTargetConfig:
    result_weight: float = 0.7
    engine_weight: float = 0.3
    cp_scale: float = 600.0
    loss_type: str = "smooth_l1"
    smooth_l1_beta: float = 0.1


@dataclasses.dataclass
class LossConfig:
    policy_weight: float = 1.0
    value_weight: float = 1.0


@dataclasses.dataclass
class OptimConfig:
    name: str = "adamw"
    lr_stem: float = 1e-5
    lr_trunk: float = 5e-5
    lr_heads: float = 2e-5
    weight_decay: float = 1e-4


@dataclasses.dataclass
class SchedulerConfig:
    name: str = "warmup_cosine"
    warmup_pct: float = 0.04
    lr_min: float = 1e-6


@dataclasses.dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclasses.dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 2
    metric: str = "val_policy_loss"


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 1
    grad_clip_norm: float = 1.0
    mixed_precision: str = "auto"  # auto | on | off
    grad_accum_steps: int = 1
    checkpoint_every_seconds: int = 300
    benchmark_every_epoch: int = 1
    min_free_disk_gb: float = 5.0


@dataclasses.dataclass
class RuntimeConfig:
    device: str = "auto"
    mode: str = "local"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclasses.dataclass
class VersionConfig:
    model_version_name: str = "v1_1"
    run_name: str = "default"


@dataclasses.dataclass
class ColabConfig:
    drive_output_dir: str = ""
    copy_data_to_local: bool = True
    local_data_dir: str = "/content/chess_alpha_data"
    auto_reduce_batch_on_oom: bool = True
    min_batch_size: int = 32
    vram_safety_margin_gb: float = 1.0
    save_every_n_steps: int = 500
    export_metrics_every_n_steps: int = 100


@dataclasses.dataclass
class TrainingRunConfig:
    paths: PathsConfig
    arch: ArchConfig = dataclasses.field(default_factory=ArchConfig)
    data_mix: DataMixConfig = dataclasses.field(default_factory=DataMixConfig)
    policy_target: PolicyTargetConfig = dataclasses.field(default_factory=PolicyTargetConfig)
    value_target: ValueTargetConfig = dataclasses.field(default_factory=ValueTargetConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    ema: EMAConfig = dataclasses.field(default_factory=EMAConfig)
    early_stopping: EarlyStoppingConfig = dataclasses.field(default_factory=EarlyStoppingConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    version: VersionConfig = dataclasses.field(default_factory=VersionConfig)
    colab: ColabConfig = dataclasses.field(default_factory=ColabConfig)


_SECTION_TYPES = {
    "paths": PathsConfig,
    "arch": ArchConfig,
    "data_mix": DataMixConfig,
    "policy_target": PolicyTargetConfig,
    "value_target": ValueTargetConfig,
    "loss": LossConfig,
    "optim": OptimConfig,
    "scheduler": SchedulerConfig,
    "ema": EMAConfig,
    "early_stopping": EarlyStoppingConfig,
    "train": TrainConfig,
    "runtime": RuntimeConfig,
    "version": VersionConfig,
    "colab": ColabConfig,
}


def load_config(path) -> TrainingRunConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuracion: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "paths" not in raw:
        raise ConfigError("La config debe tener una seccion 'paths' (checkpoint_in, vocab_path, datasets, output_dir, log_dir, validation_set_path).")

    # rutas relativas del YAML se resuelven contra la carpeta del propio
    # YAML, no contra el cwd desde donde se invoque el script - asi el
    # comportamiento no depende de si se corre desde src/, desde la raiz
    # del repo, o desde tests/.
    config_dir = path.parent.resolve()
    if isinstance(raw.get("paths"), dict):
        for key, value in raw["paths"].items():
            if isinstance(value, str) and value and not Path(value).is_absolute():
                raw["paths"][key] = str((config_dir / value).resolve())

    kwargs = {}
    for section, cls in _SECTION_TYPES.items():
        section_raw = raw.get(section, {})
        if not isinstance(section_raw, dict):
            raise ConfigError(f"La seccion '{section}' debe ser un mapping, recibi {type(section_raw)}.")
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(section_raw) - known_fields
        if unknown:
            raise ConfigError(f"Campos no soportados en '{section}': {sorted(unknown)}. Campos validos: {sorted(known_fields)}.")
        if section == "paths":
            missing = known_fields - set(section_raw)
            if missing:
                raise ConfigError(f"Faltan rutas obligatorias en 'paths': {sorted(missing)}.")
        kwargs[section] = cls(**section_raw)

    return TrainingRunConfig(**kwargs)


def _check_percentages_sum(name, parts: dict, tol=1e-6):
    total = sum(parts.values())
    if abs(total - 1.0) > tol:
        detail = ", ".join(f"{k}={v}" for k, v in parts.items())
        raise ConfigError(f"{name} debe sumar 1.0, dio {total:.6f} ({detail}).")


def _check_free_disk(path: str, min_gb: float):
    p = Path(path)
    p_check = p if p.exists() else p.parent
    while not p_check.exists() and p_check != p_check.parent:
        p_check = p_check.parent
    free_gb = shutil.disk_usage(p_check).free / (1024 ** 3)
    if free_gb < min_gb:
        raise ConfigError(f"Espacio en disco insuficiente en {p_check}: {free_gb:.1f}GB libres, se requieren {min_gb}GB.")
    return free_gb


def validate_config(cfg: TrainingRunConfig, require_gpu: bool = False, check_disk: bool = True) -> dict:
    """Corre todas las validaciones de compatibilidad. Devuelve un dict con
    info util (n_params, vocab_size, free_disk_gb, device) si todo esta OK;
    levanta ConfigError con mensaje claro en caso contrario."""
    report = {}

    # --- porcentajes ---
    _check_percentages_sum(
        "data_mix.pct_general/pct_corrective/pct_tactical",
        {"general": cfg.data_mix.pct_general, "corrective": cfg.data_mix.pct_corrective, "tactical": cfg.data_mix.pct_tactical},
    )
    _check_percentages_sum(
        "data_mix.corrective_pct_policy/value/critical",
        {"policy": cfg.data_mix.corrective_pct_policy, "value": cfg.data_mix.corrective_pct_value, "critical": cfg.data_mix.corrective_pct_critical},
    )
    if not (0.0 <= cfg.data_mix.val_fraction < 1.0):
        raise ConfigError(f"data_mix.val_fraction debe estar en [0,1), dio {cfg.data_mix.val_fraction}.")
    if cfg.data_mix.oversampling_max < 1:
        raise ConfigError(f"data_mix.oversampling_max debe ser >= 1, dio {cfg.data_mix.oversampling_max}.")
    if cfg.data_mix.general_sampler not in ("natural", "balanced"):
        raise ConfigError(f"data_mix.general_sampler '{cfg.data_mix.general_sampler}' no soportado. Opciones: natural, balanced.")
    if cfg.data_mix.balance_max_positions_per_game < 1:
        raise ConfigError(f"data_mix.balance_max_positions_per_game debe ser >= 1, dio {cfg.data_mix.balance_max_positions_per_game}.")
    if cfg.data_mix.balance_oversampling_max < 1:
        raise ConfigError(f"data_mix.balance_oversampling_max debe ser >= 1, dio {cfg.data_mix.balance_oversampling_max}.")

    # --- pesos de loss ---
    if cfg.loss.policy_weight < 0 or cfg.loss.value_weight < 0:
        raise ConfigError(f"loss.policy_weight y loss.value_weight deben ser >= 0 (dio {cfg.loss.policy_weight}, {cfg.loss.value_weight}).")
    if cfg.value_target.result_weight < 0 or cfg.value_target.engine_weight < 0:
        raise ConfigError("value_target.result_weight y engine_weight deben ser >= 0.")
    if abs((cfg.value_target.result_weight + cfg.value_target.engine_weight) - 1.0) > 1e-6:
        raise ConfigError(
            f"value_target.result_weight + engine_weight debe sumar 1.0, dio "
            f"{cfg.value_target.result_weight + cfg.value_target.engine_weight}."
        )
    if cfg.value_target.cp_scale <= 0:
        raise ConfigError(f"value_target.cp_scale debe ser > 0, dio {cfg.value_target.cp_scale}.")
    if cfg.policy_target.soft_temperature_cp <= 0:
        raise ConfigError(f"policy_target.soft_temperature_cp debe ser > 0, dio {cfg.policy_target.soft_temperature_cp}.")
    if not (0.0 <= cfg.policy_target.label_smoothing < 1.0):
        raise ConfigError(f"policy_target.label_smoothing debe estar en [0,1), dio {cfg.policy_target.label_smoothing}.")

    # --- opciones soportadas ---
    if cfg.optim.name not in SUPPORTED_OPTIMIZERS:
        raise ConfigError(f"optim.name '{cfg.optim.name}' no soportado. Opciones: {SUPPORTED_OPTIMIZERS}.")
    if cfg.scheduler.name not in SUPPORTED_SCHEDULERS:
        raise ConfigError(f"scheduler.name '{cfg.scheduler.name}' no soportado. Opciones: {SUPPORTED_SCHEDULERS}.")
    if cfg.value_target.loss_type not in SUPPORTED_VALUE_LOSSES:
        raise ConfigError(f"value_target.loss_type '{cfg.value_target.loss_type}' no soportado. Opciones: {SUPPORTED_VALUE_LOSSES}.")
    if cfg.runtime.device not in SUPPORTED_DEVICES:
        raise ConfigError(f"runtime.device '{cfg.runtime.device}' no soportado. Opciones: {SUPPORTED_DEVICES}.")
    if cfg.runtime.mode not in SUPPORTED_MODES:
        raise ConfigError(f"runtime.mode '{cfg.runtime.mode}' no soportado. Opciones: {SUPPORTED_MODES}.")
    if cfg.train.mixed_precision not in ("auto", "on", "off"):
        raise ConfigError(f"train.mixed_precision debe ser 'auto'/'on'/'off', dio {cfg.train.mixed_precision}.")
    if cfg.train.grad_accum_steps < 1:
        raise ConfigError(f"train.grad_accum_steps debe ser >= 1, dio {cfg.train.grad_accum_steps}.")

    # --- rutas ---
    checkpoint_path = Path(cfg.paths.checkpoint_in)
    vocab_path = Path(cfg.paths.vocab_path)
    if not checkpoint_path.exists():
        raise ConfigError(f"paths.checkpoint_in no existe: {checkpoint_path}")
    if not vocab_path.exists():
        raise ConfigError(f"paths.vocab_path no existe: {vocab_path}")
    for field_name in ("dataset_general", "dataset_corrective", "dataset_tactical", "unified_pool_path"):
        p = Path(getattr(cfg.paths, field_name))
        if not p.exists():
            raise ConfigError(f"paths.{field_name} no existe: {p}")

    # --- checkpoint / vocabulario / arquitectura ---
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    ckpt_vocab_size = ckpt.get("policy_vocab_size")
    if ckpt_vocab_size != len(vocab):
        raise ConfigError(
            f"Vocabulario incompatible: el checkpoint fue entrenado con policy_vocab_size={ckpt_vocab_size}, "
            f"pero {vocab_path} tiene {len(vocab)} jugadas. No se puede continuar el entrenamiento asi."
        )
    for attr in ("channels", "num_conv_blocks", "num_transformer_blocks", "num_heads"):
        ckpt_val = ckpt.get(attr)
        cfg_val = getattr(cfg.arch, attr)
        if ckpt_val != cfg_val:
            raise ConfigError(
                f"Arquitectura incompatible en '{attr}': el checkpoint tiene {ckpt_val}, la config pide {cfg_val}. "
                f"Esta fase NO debe cambiar arquitectura (ver seccion 23 del spec) - ajusta arch.{attr} en la config."
            )
    if ckpt.get("architecture") != "hybrid_cnn_transformer":
        raise ConfigError(f"El checkpoint no es de arquitectura 'hybrid_cnn_transformer' (dice: {ckpt.get('architecture')!r}).")

    report["checkpoint_epoch"] = ckpt.get("epoch")
    report["vocab_size"] = len(vocab)
    report["n_params"] = sum(v.numel() for v in ckpt["model_state"].values())

    # --- dispositivo / GPU ---
    cuda_available = torch.cuda.is_available()
    if cfg.runtime.device == "cuda" and not cuda_available:
        raise ConfigError(
            "runtime.device='cuda' pero no hay GPU disponible en este entorno (torch.cuda.is_available()==False). "
            "Esta es una condicion de parada explicita: no se debe caer a CPU en silencio para el entrenamiento completo."
        )
    if require_gpu and not cuda_available:
        raise ConfigError(
            "Se pidio validar para entrenamiento completo (require_gpu=True) pero no hay GPU disponible. "
            "Deteniendo antes de iniciar el entrenamiento completo, tal como se indico explicitamente."
        )
    resolved_device = "cuda" if (cfg.runtime.device == "auto" and cuda_available) else ("cpu" if cfg.runtime.device == "auto" else cfg.runtime.device)
    report["device"] = resolved_device
    report["cuda_available"] = cuda_available

    # --- espacio en disco ---
    if check_disk:
        report["free_disk_gb"] = _check_free_disk(cfg.paths.output_dir, cfg.train.min_free_disk_gb)

    return report


def check_tensor_shape(tensor_shape: tuple):
    if tuple(tensor_shape) != EXPECTED_TENSOR_SHAPE:
        raise ConfigError(
            f"El tensor de entrada tiene forma {tuple(tensor_shape)}, se esperaba {EXPECTED_TENSOR_SHAPE}. "
            "Esta fase tiene prohibido cambiar la representacion de entrada (ver seccion 23 del spec)."
        )
