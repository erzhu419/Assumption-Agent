# Instrument field repair, 2026-09-26

`baseline/src` freezes 0.114. `development` preserves the first field-repair implementation and its two new false completions. `development_final` preserves the repaired completion handling: 16 learning reports and 16 terminal evaluations, terminal pass 4/8 to 6/8, completed-and-pass 4/8 unchanged, no false completion. Its four medium-budget tasks increase actual relation experiments 4 to 14, distinguished outcomes 2 to 12, and supported calibrations 0 to 10. Development baseline reports are reused verbatim with original runtime and timing metadata; terminal reports are evaluated afresh.

`fresh_replication` contains 96 completed learning jobs and 96 terminal evaluations: 12 new successor apparatus, range or response change, 256/384 observation budgets, and paired 0.114/0.115 runtimes. The 12 real source libraries previously learned under 0.113 are reused unchanged; they are not new source learning. Each successor and arm starts independently from identical serialized source knowledge. Source costs are reported separately and charged equally in source-plus-successor totals. All learning finished before any terminal scoring; no formal result changed the runtime or protocol. Terminal scoring uses four fixed independent 64-row trajectories and MSE <= 1e-7. All execution is offline.

| Condition / observation budget | Terminal pass, 0.114 → 0.115 | Completed and pass | False completed |
|---|---:|---:|---:|
| Range / 256 | 5/12 → 10/12 | 4/12 → 0/12 | 0 → 0 |
| Range / 384 | 12/12 → 12/12 | 12/12 → 12/12 | 0 → 0 |
| Response / 256 | 4/12 → 8/12 | 2/12 → 0/12 | 3 → 0 |
| Response / 384 | 9/12 → 9/12 | 9/12 → 9/12 | 0 → 0 |

All terminal pass: 30/48 → 39/48. Completed-and-pass: 27/48 → 21/48. Successor learning compute decreases 2.27%, observed CPU 1.45%; observations increase 1.78%, measurements 1.53%. Medium-budget actual relation experiments increase 34 to 94, distinguished outcomes 23 to 83, supported calibrations 15 to 75. Raw group costs and paired wins/losses are in `fresh_replication/summary.json`; mechanism counts and remaining actions are in `fresh_replication/mechanism_diagnosis.json`.

Use `scripts/run_boundary_method_transfer_study.py --phase {learn,evaluate,summary} --output artifacts/instrument_field_repair_20260926/{development,development_final,fresh_replication} --arm {baseline,revised}`. Summary does not need an arm. Frozen protocols contain runtime/source paths, seeds, budgets and worker count.

## Limitations

All 24 revised medium-budget cases remain unresolved, including 18 that pass terminal evaluation. The completion rule is more conservative. Response-change tasks have two paired MSE regressions at each budget, despite unchanged high-budget pass count. The 192 budget, multi-generation knowledge accumulation, ordinary Code Agent, and external LLM baselines are not evaluated. `next_gap.json` separates remaining single-task planning failures from the next mainline: knowledge handoff, later reuse and withdrawal, and cumulative research value across successors.
