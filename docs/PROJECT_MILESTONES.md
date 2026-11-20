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
| August 2026 | Frozen Sequence Policy test | Test design and integrity checks prepared for 50,016 windows / 4,118,551 targets; captured log alone does not establish final results |
| August 2026 | Search engine phases | Alpha–Beta/PVS/TT/quiescence/LMR/null move/DFPN-PNS plus correctness and tactical test cases |
| August 2026 | Enrichment V3 | 28 feature families completed over 818,611 rows, 9/9 shards, 100% join, 30/30 tests; bitboard dtype issue corrected |
| August–September 2026 | Search observability / Search Tree v1.1 | 458 passed + 1 skipped; 450 benchmark runs and no observed illegal moves/state mutation |
| August–September 2026 | Search-aware feature design | Proposal developed from 10 → 40 → ~60 candidate signals; final full-run and impact selection not established |
| 2026 experimental calibration | Policy-prior λ=0.15 | Fase 6 improved ranking metrics on 703 positions; Fase 7 rejected after critical mate-defense and discovered-attack regressions |
| Future research | Self-play / MCTS and additional neural architectures | Discussed as prospective direction, not represented as completed implementation |

## Interpretation rules

1. **Different datasets, different metrics.** The million-game corpus, 10k golden sample, 818,611 position rows, and 703-position tuning gate are different evaluation populations.
2. **Different models.** The board policy/value system and causal Sequence Policy transformer are not interchangeable.
3. **Controlled Elo only.** The reported +96.2 Elo belongs to a single comparison against `epoch5` with specific hybrid search settings.
4. **Negative results count.** A promising λ-tuning phase did not override the subsequent tactical rejection gate.
5. **Snapshots are not additive.** Search test counts from different milestones (e.g., 387 and 458) are sequential suite snapshots, not separate tests to total together.
6. **Reproduction requires manifests.** Archived summaries are leads; code and machine-generated reports should be source of truth for final public claims.