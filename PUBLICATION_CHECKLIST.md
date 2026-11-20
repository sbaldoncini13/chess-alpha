# GitHub publication checklist — Chess Alpha

Status of the publication checklist after the source-code audit and final pre-publication review of 2026-11-20.

## Evidence and accuracy

- [x] README compared with the actual source tree and module names. It now describes only published code, and lists unpublished components explicitly.
- [x] 1,000,000 games / 81,244,568 move tokens confirmed from `tokenization_v1/outputs/full_1000000/manifest.json` (seed 20260803, SHA-256 per file).
- [x] Golden 10k / 808,611 plies / 818,611 position rows confirmed from manifests.
- [x] Sequence Policy FULL architecture confirmed from `configs/model_base.yaml` (8 layers, d=384, 6 heads, FFN 1536, ctx 256, vocab 20,483).
- [x] `sequence_policy_full_test_2286.json` located (`FROZEN_TEST_FINAL`). Final test metrics are now reported.
- [x] Search Tree V1 test suite re-run on the exported tree (see README → Tests).
- [x] Search-aware 60-family status verified: implemented and piloted, full production **not** completed. Reported as experimental.
- [x] Negative λ=0.15 result preserved, with figures checked against the phase-6/phase-7 summaries (703 positions; 568-position tactical gate).
- [x] Experimental and unpublished branches listed (Relative2D, separate towers, search_enrichment_v1, tree_optimization_v1, arena pilot).
- [x] The 28 V3 feature-family names in `docs/ARCHITECTURE.md` checked against `tokenization_enrichment_v3/scripts/feature_registry.py`.

## Public safety and repository hygiene

- [x] No API keys, tokens, credentials or `.env` files in the exported tree (scanned).
- [x] No personal filesystem paths in exported files (scanned).
- [x] No datasets, PGN files, checkpoints, archives, binaries, logs or notebooks in the export. `.gitignore` blocks them.
- [x] No Stockfish code or binary included. Stockfish is only called as an external UCI process, through python-chess, by optional scripts.
- [x] Every exported source file is byte-identical to the research tree (`EXPORT_MANIFEST.sha256`).
- [x] License: GPL-3.0-or-later for the project's own code (`LICENSE`, the canonical FSF text). Third-party terms are listed in `THIRD_PARTY_NOTICES.md`. No third-party code is bundled.
- [x] Datasets and model checkpoints are excluded from this release. Publishing any of them later requires confirming the redistribution rights of every PGN source first.
- [ ] Optional: add CI (GitHub Actions) running `pytest` with `requirements.txt` on Linux.

## Reproducibility

- [x] `requirements.txt` (core) and `requirements-optional.txt` (torch CPU, duckdb) with validated versions.
- [x] Two runnable examples on synthetic positions (`examples/`).
- [x] Tests that need private data or checkpoints skip cleanly or are left out of collection (`conftest.py`).
- [ ] Tests were validated on Windows / Python 3.12 only. They have not been run on Linux or macOS.
