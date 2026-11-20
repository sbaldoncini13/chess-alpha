"""Correccion 5: tests con archivos pequenos/sinteticos (NUNCA sobre los
shards reales de test_10000 -- eso se hace en el script 13 aparte)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from shard_integrity import (  # noqa: E402
    build_integrity_manifest, sha256_of_file, verify_shard_integrity,
)


def _make_shard(tmp_path, tag, content_positions=b"fake-parquet-bytes",
                 content_bin=b"fake-bin-bytes", content_off=b"fake-offsets-bytes"):
    (tmp_path / f"{tag}.positions.parquet").write_bytes(content_positions)
    (tmp_path / f"{tag}.legalmoves.bin").write_bytes(content_bin)
    (tmp_path / f"{tag}.legalmoves_offsets.npy").write_bytes(content_off)


def test_sha256_of_file_deterministic(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert sha256_of_file(str(p)) == sha256_of_file(str(p))
    assert sha256_of_file(str(p)) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )  # sha256("hello world"), verificado con hashlib directamente


def test_build_integrity_manifest_covers_all_shards_and_files(tmp_path):
    _make_shard(tmp_path, "shard_00000")
    _make_shard(tmp_path, "shard_00001")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    assert manifest["n_shards"] == 2
    assert manifest["algorithm"] == "sha256"
    assert set(manifest["shards"].keys()) == {"shard_00000", "shard_00001"}
    for tag, files in manifest["shards"].items():
        assert set(files.keys()) == {
            f"{tag}.positions.parquet", f"{tag}.legalmoves.bin", f"{tag}.legalmoves_offsets.npy",
        }
        for info in files.values():
            assert "sha256" in info and "size_bytes" in info


def test_verify_pass_on_untouched_shards(tmp_path):
    _make_shard(tmp_path, "shard_00000")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    result = verify_shard_integrity(str(tmp_path), manifest)
    assert result["status"] == "PASS"
    assert result["failures"] == []


def test_verify_fails_on_missing_file(tmp_path):
    _make_shard(tmp_path, "shard_00000")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    os.remove(tmp_path / "shard_00000.legalmoves.bin")
    result = verify_shard_integrity(str(tmp_path), manifest)
    assert result["status"] == "FAIL"
    reasons = {f["reason"] for f in result["failures"]}
    assert "archivo ausente" in reasons


def test_verify_fails_on_size_mismatch(tmp_path):
    _make_shard(tmp_path, "shard_00000")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    (tmp_path / "shard_00000.positions.parquet").write_bytes(b"different-length-content-here")
    result = verify_shard_integrity(str(tmp_path), manifest)
    assert result["status"] == "FAIL"
    reasons = {f["reason"] for f in result["failures"]}
    assert "tamano distinto" in reasons


def test_verify_fails_on_hash_mismatch_same_size(tmp_path):
    _make_shard(tmp_path, "shard_00000", content_positions=b"12345678")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    # mismo tamano (8 bytes), contenido distinto -> mismo size_bytes, hash distinto
    (tmp_path / "shard_00000.positions.parquet").write_bytes(b"87654321")
    result = verify_shard_integrity(str(tmp_path), manifest)
    assert result["status"] == "FAIL"
    reasons = {f["reason"] for f in result["failures"]}
    assert "sha256 incorrecto" in reasons


def test_verify_fails_on_manifest_shard_not_on_disk(tmp_path):
    _make_shard(tmp_path, "shard_00000")
    manifest = build_integrity_manifest(str(tmp_path), "test_synth")
    _make_shard(tmp_path, "shard_00001")  # shard nuevo, no estaba en el manifest
    result = verify_shard_integrity(str(tmp_path), manifest)
    assert result["status"] == "FAIL"
    reasons = {f["reason"] for f in result["failures"]}
    assert "shard presente en disco pero ausente en el manifest de integridad" in reasons


def test_manifest_does_not_modify_source_files(tmp_path):
    _make_shard(tmp_path, "shard_00000", content_positions=b"original-content")
    before = (tmp_path / "shard_00000.positions.parquet").read_bytes()
    build_integrity_manifest(str(tmp_path), "test_synth")
    after = (tmp_path / "shard_00000.positions.parquet").read_bytes()
    assert before == after
