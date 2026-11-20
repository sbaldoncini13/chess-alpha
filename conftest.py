"""Repository-level pytest configuration (added for the public export).

Two test modules validate the code against the project's private datasets.
Those datasets are not distributed, so the modules are only collected when
the data is present locally. No test file was modified.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

_DATA_DEPENDENT_TESTS = {
    # Round-trip FEN checks over a sample of the 40k-position search dataset.
    "tests/search_v1/test_roundtrip.py": ROOT / "data" / "processed" / "search_tree_v1" / "final" / "positions.parquet",
    # Smoke test over ~100 real rows of the frozen 10k-game golden dataset.
    "tokenization_enrichment_v3/tests/test_smoke_100_golden.py": ROOT / "tokenization_v1" / "shards_v2" / "test_10000",
}

collect_ignore = [test for test, required in _DATA_DEPENDENT_TESTS.items() if not required.exists()]
