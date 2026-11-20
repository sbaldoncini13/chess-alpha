"""Proteccion minima contra sobrescritura accidental de un Golden Dataset
congelado. Usado por 06_enrich_positions.py y 16_stockfish_full_enrichment.py
(los dos unicos scripts capaces de regenerar shards_v2/<test_name>/ u
outputs_v2/<test_name>/stockfish_full/). Deliberadamente pequeno: una
funcion de busqueda + una funcion de enforcement, sin tocar la arquitectura
de ninguno de los dos pipelines."""
import json
import os


def find_golden_lock_for_test(golden_root: str, test_name: str):
    """Busca, entre todos los subdirectorios de golden_root, un
    GOLDEN_DATASET.lock cuyo test_name coincida. Devuelve (lock_path, lock)
    o None si no hay ninguno protegiendo ese test_name."""
    if not os.path.isdir(golden_root):
        return None
    for entry in sorted(os.listdir(golden_root)):
        lock_path = os.path.join(golden_root, entry, "GOLDEN_DATASET.lock")
        if not os.path.isfile(lock_path):
            continue
        try:
            with open(lock_path, encoding="utf-8") as f:
                lock = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if lock.get("test_name") == test_name:
            return lock_path, lock
    return None


def enforce_golden_lock(golden_root: str, test_name: str, allow_overwrite: bool, logger) -> None:
    """Si test_name esta protegido por un lock, se detiene (logger.log +
    SystemExit) salvo allow_overwrite=True (en cuyo caso solo advierte).
    No hace nada si no hay lock para ese test_name."""
    found = find_golden_lock_for_test(golden_root, test_name)
    if found is None:
        return
    lock_path, lock = found
    dataset_id = lock.get("golden_dataset_id", "?")
    version = lock.get("golden_dataset_version", "?")
    if allow_overwrite:
        logger.log(
            f"ADVERTENCIA: --allow-golden-overwrite activo -- se IGNORA el lock en {lock_path} "
            f"(golden_dataset_id={dataset_id}, version={version}). Esto puede invalidar un "
            f"Golden Dataset congelado y auditado."
        )
        return
    logger.log(
        f"ERROR: '{test_name}' esta protegido por un GOLDEN_DATASET.lock en {lock_path} "
        f"(golden_dataset_id={dataset_id}, version={version}). Este pipeline NO puede "
        f"sobrescribir un Golden Dataset congelado. Usar --allow-golden-overwrite solo si "
        f"se sabe exactamente lo que se esta haciendo (esto invalida la version congelada)."
    )
    raise SystemExit(1)
