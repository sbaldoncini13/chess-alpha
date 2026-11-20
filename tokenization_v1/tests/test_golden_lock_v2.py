import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from golden_lock import enforce_golden_lock, find_golden_lock_for_test  # noqa: E402


class ListLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


def _write_lock(golden_root, subdir, test_name, dataset_id="ds1", version="1.0.0"):
    d = os.path.join(golden_root, subdir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "GOLDEN_DATASET.lock"), "w", encoding="utf-8") as f:
        json.dump({"golden_dataset_id": dataset_id, "golden_dataset_version": version,
                    "test_name": test_name, "manifest_sha256": "abc"}, f)


def test_find_returns_none_when_golden_root_missing(tmp_path):
    assert find_golden_lock_for_test(str(tmp_path / "nope"), "test_10000") is None


def test_find_returns_none_when_no_lock_matches_test_name(tmp_path):
    _write_lock(str(tmp_path), "other_v1", "other_test")
    assert find_golden_lock_for_test(str(tmp_path), "test_10000") is None


def test_find_returns_lock_when_test_name_matches(tmp_path):
    _write_lock(str(tmp_path), "test_10000_v1", "test_10000")
    found = find_golden_lock_for_test(str(tmp_path), "test_10000")
    assert found is not None
    lock_path, lock = found
    assert lock["test_name"] == "test_10000"


def test_find_ignores_corrupt_lock_file(tmp_path):
    d = os.path.join(str(tmp_path), "broken_v1")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "GOLDEN_DATASET.lock"), "w", encoding="utf-8") as f:
        f.write("not json")
    assert find_golden_lock_for_test(str(tmp_path), "test_10000") is None


def test_enforce_noop_when_no_lock(tmp_path):
    logger = ListLogger()
    enforce_golden_lock(str(tmp_path), "test_10000", allow_overwrite=False, logger=logger)
    assert logger.lines == []


def test_enforce_blocks_when_locked_and_no_override(tmp_path):
    _write_lock(str(tmp_path), "test_10000_v1", "test_10000")
    logger = ListLogger()
    with pytest.raises(SystemExit):
        enforce_golden_lock(str(tmp_path), "test_10000", allow_overwrite=False, logger=logger)
    assert any("esta protegido" in line for line in logger.lines)


def test_enforce_allows_with_explicit_override_but_warns(tmp_path):
    _write_lock(str(tmp_path), "test_10000_v1", "test_10000")
    logger = ListLogger()
    enforce_golden_lock(str(tmp_path), "test_10000", allow_overwrite=True, logger=logger)  # no debe lanzar
    assert any("ADVERTENCIA" in line for line in logger.lines)


def test_enforce_does_not_block_unrelated_test_name(tmp_path):
    _write_lock(str(tmp_path), "test_10000_v1", "test_10000")
    logger = ListLogger()
    enforce_golden_lock(str(tmp_path), "smoke_20", allow_overwrite=False, logger=logger)  # no debe lanzar
    assert logger.lines == []
