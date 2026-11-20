# Chess Alpha — Milestones and decision log

This is a **historical milestone summary**, not a literal chronological Git log. Dates represent the period in which work was discussed or recorded, not necessarily commit timestamps.

| Period | Milestone | Outcome / decision |
|---|---|---|
| July 2026 | Chess Alpha v1.1 policy/value evaluation | `control_epoch02` selected; `balanced_epoch02` archived; controlled arena vs `epoch5` reported 63.5% score over 100 games, +96.2 relative Elo |
| July 2026 | Dataset diversification | Candidate selection reviewed; correct limit established per full ECO code rather than broad A–E group |
| Late July 2026 | Relative2D experiment | A/B alternatives documented; comparative coverage, value, CPL and latency validation still needed |
| Early August 2026 | PGN tokenizer | Original 4.47M-game source inspected; uint16 action IDs, deterministic sampling and restartable pipeline developed |
| August 2026 | 10k pilot | 10,000 unique valid games, 808,611 plies, 25/25 tokenizer tests, real stop/resume verification |
| August 2026 | Main million-game corpus | Project records report 1M selected games and ~81.2M plies; preserve original manifest for artifact-level verification |
| August 2026 | Golden Dataset | 10,000-game immutable subset, 818,611 position-level rows, nine shards, SHA-256 and structural checks |
| August 2026 | Sequence Policy FULL | Causal transformer checkpoint recovered at step 2286 / 74,536,785 tokens seen; earlier gradient accumulation issue corrected |
| August 2026 | Frozen Sequence Policy test | Read once (`FROZEN_TEST_FINAL`) on 50,016 windows / 4,118,551 targets: top-1 0.301, top-10 0.701, MRR 0.434, which matches dev (top-1 0.301, MRR 0.434) |
| August 2026 | Search engine phases | Alpha–Beta/PVS/TT/quiescence/LMR/null move/DFPN-PNS plus correctness and tactical test cases |
| August 2026 | Enrichment V3 | 28 feature families completed over 818,611 rows, 9/9 shards, 100% join, 30/30 tests; bitboard dtype issue corrected |
| August 2026 | Epoch4 selection | 500-game confirmatory arena vs `control_epoch02`: Epoch4 scored 60.8 %, with an Elo CI above 0 and 0 errors/illegal moves. Promoted to the operational model |
| August 2026 | Search Tree V1 phases VII–VIII | Epoch4 (reorder-only, `root_only`) and Sequence Policy + RRF fusion (opt-in) integrated. Closure suite: 458 passed + 1 skipped. N=2000 benchmark: 0 illegal moves in 11,988 searches. Fusion did not beat Epoch4 alone, so it stays off by default |
| August 2026 | Search observability (Search Tree Observatory) | Opt-in per-node tracing. 90 positions × 5 configurations = 450 runs with no observed illegal moves or state mutation |
| August 2026 | Search-aware feature enrichment | 60-family implementation. 1k and 10k pilots completed. The 100k gate ran 2,000/2,000 chunks but closed with operational PASS = FALSE (throughput, Windows standby). The full production run was not completed |
| August 2026 | Search-parameter tuning and arena pilot | Random Search / Optuna tuning with safety gates and a 40-game `search_v1` arena pilot (50 % score, inconclusive). Both experimental and unpublished |
| 2026 experimental calibration | Policy-prior λ=0.15 | Fase 6 improved ranking metrics on 703 positions; Fase 7 rejected after critical mate-defense and discovered-attack regressions |
| Future research | Self-play / MCTS and additional neural architectures | Discussed as prospective direction, not represented as completed implementation |
| November 2026 | Source-code publication audit | Code inventoried and audited for secrets and provenance. Core packages exported byte-identically, with test suites re-run on the export (see README) |

## Interpretation rules

1. **Different datasets, different metrics.** The million-game corpus, 10k golden sample, 818,611 position rows, and 703-position tuning gate are different evaluation populations.
2. **Different models.** The board policy/value system and causal Sequence Policy transformer are not interchangeable.
3. **Controlled Elo only.** The reported +96.2 Elo belongs to a single comparison against `epoch5` with specific hybrid search settings.
4. **Negative results count.** A promising λ-tuning phase did not override the subsequent tactical rejection gate.
5. **Snapshots are not additive.** Search test counts from different milestones (e.g., 387 and 458) are sequential suite snapshots, not separate tests to total together.
6. **Reproduction requires manifests.** Archived summaries are leads; code and machine-generated reports should be source of truth for final public claims.