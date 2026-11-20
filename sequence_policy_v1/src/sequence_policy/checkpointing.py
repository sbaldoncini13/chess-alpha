"""Checkpoint completo (modelo, optimizer, scheduler, scaler, sampler, RNGs,
config, contadores) con escritura atomica (seccion 15/22 de la conversacion
de Colab, reusado ya en Etapa D para el smoke local)."""
import os
import random

import numpy as np
import torch


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                     sampler_state=None, global_step=0, tokens_seen=0, epoch_equivalent=0.0,
                     config=None, dataset_hashes=None, split_hashes=None, best_metrics=None,
                     extra=None):
    import time
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "sampler_state": sampler_state,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "epoch_equivalent": epoch_equivalent,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
        "dataset_hashes": dataset_hashes,
        "split_hashes": split_hashes,
        "best_metrics": best_metrics,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    with open(tmp, "rb") as f:
        f.read(64)  # verifica que el archivo es legible antes de promoverlo
    os.replace(tmp, path)
    return payload


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                     expected_config=None, expected_dataset_hashes=None, expected_split_hashes=None,
                     strict_compat=True, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if strict_compat:
        mismatches = []
        if expected_config is not None and ckpt.get("config") != expected_config:
            mismatches.append("config")
        if expected_dataset_hashes is not None and ckpt.get("dataset_hashes") != expected_dataset_hashes:
            mismatches.append("dataset_hashes")
        if expected_split_hashes is not None and ckpt.get("split_hashes") != expected_split_hashes:
            mismatches.append("split_hashes")
        if mismatches:
            raise ValueError(
                f"Checkpoint incompatible en {mismatches}: no se carga parcialmente, "
                f"se detiene (ver seccion 22/26 -- no reparar automaticamente)."
            )

    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch_cpu"])
    if ckpt.get("rng_torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])

    return ckpt
