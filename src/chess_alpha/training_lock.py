"""Lock file exclusivo para evitar que dos corridas de entrenamiento (o una
corrida y un benchmark pesado) escriban en la misma ruta de salida al mismo
tiempo - motivado por la colision real de archivos detectada entre esta
sesion y una sesion paralela en reports/v2_relative2d/ durante la rama
Relative2D. Guarda PID, hostname, timestamp, hash de config y hash de
dataset; rechaza adquirir el lock si ya existe uno de un proceso VIVO."""

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


class TrainingLockError(RuntimeError):
    """Ya existe un lock activo de otro proceso - no se debe entrenar encima."""


def _pid_is_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


@contextmanager
def training_lock(lock_path: Path, config_hash: str, dataset_hash: str, extra: dict = None):
    """Uso:
        with training_lock(lock_path, config_hash, dataset_hash):
            ... cuerpo del entrenamiento ...

    Si el lock ya existe y el PID que lo escribio sigue vivo, lanza
    TrainingLockError sin tocar nada. Si el PID ya no existe (lock viejo de
    una corrida que crasheo sin limpiar), lo reporta como "stale" y lo
    reemplaza. Al salir normalmente borra el lock; si el cuerpo lanza una
    excepcion, el lock queda escrito con status="error" (para diagnostico)
    en vez de borrarse."""
    lock_path = Path(lock_path)
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing_pid = existing.get("pid")
        if existing_pid is not None and _pid_is_alive(existing_pid):
            raise TrainingLockError(
                f"Lock activo encontrado en {lock_path}: PID={existing_pid} host={existing.get('hostname')} "
                f"desde {existing.get('started_at')} (status={existing.get('status')}). "
                "Otro proceso parece estar entrenando/usando esta ruta - no se inicia esta corrida."
            )
        print(f"[training_lock] Lock existente en {lock_path} pertenece a un PID que ya no esta vivo "
              f"(PID={existing_pid}) - se considera obsoleto (stale) y se reemplaza.")

    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "status": "running",
    }
    if extra:
        payload.update(extra)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[training_lock] Lock adquirido: {lock_path} (PID={payload['pid']})")

    try:
        yield payload
    except BaseException as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)
        payload["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[training_lock] Corrida termino con error - lock conservado con status=error para diagnostico: {lock_path}")
        raise
    else:
        lock_path.unlink(missing_ok=True)
        print(f"[training_lock] Lock liberado correctamente: {lock_path}")
