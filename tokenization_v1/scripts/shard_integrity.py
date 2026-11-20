"""Correccion 5 de la auditoria: checksums SHA-256 de integridad para los
shards YA ESCRITOS de shards_v2/<test_name>/. Puramente de lectura sobre
archivos existentes -- nunca regenera ni modifica ningun shard.

Se guarda en un archivo APARTE (`integrity_manifest.json`), nunca dentro de
`manifest.json` (que se preserva intacto como artefacto historico del
piloto, segun lo pedido explicitamente)."""
import hashlib
import json
import os
import time

CHECKSUM_ALGORITHM = "sha256"
COVERED_EXTENSIONS = (".positions.parquet", ".legalmoves.bin", ".legalmoves_offsets.npy")
_CHUNK_SIZE = 1024 * 1024


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _shard_ids(shard_dir: str) -> list:
    ids = sorted({
        f.split(".")[0] for f in os.listdir(shard_dir) if f.endswith(".positions.parquet")
    })
    return ids


def compute_shard_checksums(shard_dir: str) -> dict:
    """Devuelve {shard_tag: {filename: {size_bytes, sha256}}} para los 3
    archivos fundamentales de cada shard ya presente en `shard_dir`."""
    result = {}
    for tag in _shard_ids(shard_dir):
        files = {}
        for ext in COVERED_EXTENSIONS:
            fname = f"{tag}{ext}"
            path = os.path.join(shard_dir, fname)
            if not os.path.isfile(path):
                files[fname] = {"missing": True}
                continue
            files[fname] = {
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_of_file(path),
            }
        result[tag] = files
    return result


def build_integrity_manifest(shard_dir: str, test_name: str) -> dict:
    checksums = compute_shard_checksums(shard_dir)
    return {
        "test_name": test_name,
        "algorithm": CHECKSUM_ALGORITHM,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_shards": len(checksums),
        "covered_extensions": list(COVERED_EXTENSIONS),
        "shards": checksums,
        "note": (
            "Checksums calculados sobre archivos ya existentes de "
            "shards_v2/<test_name>/, sin regenerar ni modificar ningun "
            "shard. Archivo separado de manifest.json (preservado intacto)."
        ),
    }


def verify_shard_integrity(shard_dir: str, integrity_manifest: dict) -> dict:
    """PASS/FAIL por shard/archivo. Detecta: archivo ausente, tamano
    distinto, hash incorrecto, entrada de manifest inconsistente (shard en
    disco sin entrada en el manifest, o viceversa)."""
    failures = []
    manifest_shards = integrity_manifest.get("shards", {})
    disk_shard_ids = set(_shard_ids(shard_dir))
    manifest_shard_ids = set(manifest_shards.keys())

    for missing_in_manifest in disk_shard_ids - manifest_shard_ids:
        failures.append({"shard": missing_in_manifest, "file": None,
                          "reason": "shard presente en disco pero ausente en el manifest de integridad"})
    for missing_on_disk in manifest_shard_ids - disk_shard_ids:
        failures.append({"shard": missing_on_disk, "file": None,
                          "reason": "shard presente en el manifest pero ausente en disco"})

    for tag, files in manifest_shards.items():
        for fname, expected in files.items():
            path = os.path.join(shard_dir, fname)
            if expected.get("missing"):
                if os.path.isfile(path):
                    failures.append({"shard": tag, "file": fname,
                                      "reason": "el manifest registraba el archivo como ausente pero ahora existe"})
                continue
            if not os.path.isfile(path):
                failures.append({"shard": tag, "file": fname, "reason": "archivo ausente"})
                continue
            actual_size = os.path.getsize(path)
            if actual_size != expected["size_bytes"]:
                failures.append({
                    "shard": tag, "file": fname, "reason": "tamano distinto",
                    "expected_size_bytes": expected["size_bytes"], "actual_size_bytes": actual_size,
                })
                continue
            actual_hash = sha256_of_file(path)
            if actual_hash != expected["sha256"]:
                failures.append({
                    "shard": tag, "file": fname, "reason": "sha256 incorrecto",
                    "expected_sha256": expected["sha256"], "actual_sha256": actual_hash,
                })

    return {
        "status": "PASS" if not failures else "FAIL",
        "n_shards_checked": len(manifest_shards),
        "n_files_checked": sum(len(f) for f in manifest_shards.values()),
        "failures": failures,
    }


def load_integrity_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
