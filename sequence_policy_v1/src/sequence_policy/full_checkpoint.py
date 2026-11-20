"""Schema de checkpoint del FULL TRAINING -- envuelve checkpointing.py
(save_checkpoint/load_checkpoint, ya probado en Etapa D) sin modificarlo,
agregando la metadata especifica de esta corrida via el dict `extra` que
checkpointing.py ya soporta de forma generica.

Distincion explicita de contadores (obligatoria por el pedido del usuario):
  - run_global_step / run_tokens_seen: SOLO de esta corrida (full training),
    arrancan en 0.
  - pilot_global_step_reference / pilot_tokens_seen_reference: metadata
    HISTORICA del piloto, nunca se incrementan aca.
  - cumulative_tokens_seen_including_pilot: run_tokens_seen + pilot_tokens_seen_reference,
    solo informativo -- NUNCA se usa como contador operativo real (no hay
    continuidad real de sampler entre piloto y full, ver WEIGHTS_WARM_START_FROM_PILOT_FULL_NEW_EPOCH).
"""
import os
import time

from sequence_policy.checkpointing import load_checkpoint as _load_checkpoint_base
from sequence_policy.checkpointing import save_checkpoint as _save_checkpoint_base

SCHEMA_VERSION = "full_training_v1"


def save_full_checkpoint(path, model, optimizer, scheduler, scaler, sampler,
                          run_global_step, run_tokens_seen, epoch_equivalent,
                          model_config, dataset_hashes, split_hashes,
                          pilot_checkpoint_sha256, pilot_global_step_reference, pilot_tokens_seen_reference,
                          warm_start_source, code_version, dev_metrics_recent=None, run_config=None):
    current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None
    cumulative = run_tokens_seen + pilot_tokens_seen_reference

    extra = {
        "schema_version": SCHEMA_VERSION,
        "run_name": "sequence_policy_base_full_v1",
        "classification": "WEIGHTS_WARM_START_FROM_PILOT_FULL_NEW_EPOCH",
        "run_global_step": run_global_step,
        "run_tokens_seen": run_tokens_seen,
        "cumulative_tokens_seen_including_pilot": cumulative,
        "pilot_checkpoint_sha256": pilot_checkpoint_sha256,
        "pilot_global_step_reference": pilot_global_step_reference,
        "pilot_tokens_seen_reference": pilot_tokens_seen_reference,
        "warm_start_source": warm_start_source,
        "warm_start_note": "Solo model_state_dict provino del piloto. optimizer/scheduler/scaler/sampler "
                            "son NUEVOS de esta corrida -- no es FULL_STATE_RESUME.",
        "current_lr": current_lr,
        "code_version": code_version,
        "dev_metrics_recent": dev_metrics_recent or {},
        "run_config": run_config or {},
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    sampler_state = sampler.state_dict() if sampler is not None else None

    payload = _save_checkpoint_base(
        path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        sampler_state=sampler_state, global_step=run_global_step, tokens_seen=run_tokens_seen,
        epoch_equivalent=epoch_equivalent, config={"model": model_config},
        dataset_hashes=dataset_hashes, split_hashes=split_hashes, extra=extra,
    )
    return payload


def load_full_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                          expected_model_config=None, expected_dataset_hashes=None,
                          expected_split_hashes=None, strict_compat=True, map_location="cpu"):
    expected_config = {"model": expected_model_config} if expected_model_config is not None else None
    ckpt = _load_checkpoint_base(
        path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        expected_config=expected_config, expected_dataset_hashes=expected_dataset_hashes,
        expected_split_hashes=expected_split_hashes, strict_compat=strict_compat, map_location=map_location,
    )
    extra = ckpt.get("extra", {})
    if extra.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint no tiene el schema esperado ('{SCHEMA_VERSION}'), encontrado: "
            f"'{extra.get('schema_version')}'. No se repara automaticamente -- puede ser un checkpoint "
            f"del piloto (schema distinto) cargado por error con esta funcion en vez de warm_start_from_pilot()."
        )
    return ckpt


def warm_start_from_pilot(pilot_checkpoint_path, model, expected_pilot_model_config=None):
    """Carga UNICAMENTE los pesos del modelo desde el checkpoint del piloto.
    No toca optimizer/scheduler/scaler/sampler -- esos se crean nuevos fuera
    de esta funcion. Devuelve la metadata de procedencia para guardarla en
    el primer checkpoint del full training."""
    import hashlib

    import torch

    if not os.path.isfile(pilot_checkpoint_path):
        raise FileNotFoundError(f"No existe el checkpoint del piloto: {pilot_checkpoint_path}")

    h = hashlib.sha256()
    with open(pilot_checkpoint_path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    pilot_sha256 = h.hexdigest()

    ckpt = torch.load(pilot_checkpoint_path, map_location="cpu", weights_only=False)
    pilot_model_cfg = ckpt["config"]["model"]
    if expected_pilot_model_config is not None and pilot_model_cfg != expected_pilot_model_config:
        raise ValueError(
            f"Arquitectura del checkpoint del piloto no coincide con la esperada para el full "
            f"training. Piloto: {pilot_model_cfg}. Esperada: {expected_pilot_model_config}. "
            f"No se cambia la arquitectura BASE silenciosamente -- se detiene."
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
    if missing or unexpected:
        raise ValueError(f"load_state_dict con claves faltantes/inesperadas: missing={missing}, "
                          f"unexpected={unexpected}. No se continua con una carga parcial.")

    # verificacion de integridad numerica de los pesos cargados (no confiar ciegamente)
    n_nan, n_inf = 0, 0
    for p in model.parameters():
        n_nan += int(torch.isnan(p).sum().item())
        n_inf += int(torch.isinf(p).sum().item())
    if n_nan > 0 or n_inf > 0:
        raise ValueError(f"Pesos del piloto contienen NaN/Inf tras cargar (n_nan={n_nan}, n_inf={n_inf}). "
                          f"No se continua un warm-start desde pesos corruptos.")

    return {
        "pilot_checkpoint_sha256": pilot_sha256,
        "pilot_global_step_reference": ckpt.get("global_step"),
        "pilot_tokens_seen_reference": ckpt.get("tokens_seen"),
        "pilot_model_config": pilot_model_cfg,
        "warm_start_source": os.path.abspath(pilot_checkpoint_path),
        "warm_start_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
