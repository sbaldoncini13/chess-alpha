# GitHub publication checklist — Chess Alpha

This checklist helps transform a historical technical summary into a trustworthy public software repository.

## Evidence and accuracy

- [ ] Compare this README with the *current* Git working tree, branches and module names.
- [ ] Confirm the 1,000,000-game / ~81.2M-ply count from the final run's manifest.
- [ ] Confirm Golden 10k / 808,611 plies / 818,611 position rows / nine shards, plus SHA-256 manifests.
- [ ] Verify all 28 V3 feature-family names from actual schemas or data dictionaries; do not infer names.
- [ ] Confirm the Search Tree v1.1 test suite output and observability benchmark report.
- [ ] Inspect exact Sequence Policy FULL checkpoint config and training logs.
- [ ] Locate `sequence_policy_full_test_2286.json` before claiming definitive frozen-test metrics.
- [ ] Verify search-aware 40–60 feature generation status and any trained downstream decision-tree model before marking them complete.
- [ ] Preserve the negative tactical decision for λ=0.15 and the gate that rejected promotion.
- [ ] Verify which branches were merged, archived, or remain experimental.

## Public safety and repository hygiene

- [ ] Exclude API keys, tokens, SSH keys, credentials, `.env` files and machine-specific secrets.
- [ ] Remove personal filesystem paths from code examples and configuration defaults.
- [ ] Check all PGN source and derived-data redistribution rights.
- [ ] Check licensing/redistribution for Stockfish and other third-party binaries; include required notices where applicable.
- [ ] Do not push massive corpora, backups, checkpoints or archives inadvertently; consider Git LFS or independent artifact hosting for permitted large files.
- [ ] Exclude `.venv/`, cache directories, local logs, temporary shards and machine-generated intermediates from normal commits unless needed for a reproducible small example.
- [ ] Add a clear open-source license **only after explicitly choosing one**.
- [ ] Record Python/dependency versions, installation instructions and test commands from the actual project.
- [ ] Provide at least one small publicly redistributable example demonstrating tokenization and legality.
- [ ] Where possible, add CI for tokenizer, FEN, move legality, feature joins and search regression tests.

## Suggested repository layout (adapt to what exists)

```text
chess-alpha/
├── README.md
├── PUBLICATION_CHECKLIST.md
├── docs/
│   ├── TECHNICAL_OVERVIEW_ES.md
│   └── PROJECT_MILESTONES.md
├── tokenization_v1/               # Only if this directory exists
├── sequence_policy_v1/             # Only if this directory exists
├── search_tree_v1/                 # Use actual names from repo
├── [other existing source modules]
└── [tests and reproducible examples]
```

The documentation files in this package are intended to sit at the repository root (and under `docs/`). **Do not create empty implementation folders just to match the illustration.**