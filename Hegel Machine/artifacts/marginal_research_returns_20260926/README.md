# Marginal research returns

Completed offline comparison: 0.119.1 baseline, final 0.120 and empty-start cumulative research, on 12 predeclared new 9987-series apparatus. All 36 sessions completed **216 first visits + 38 revisits**, then all 216 initial/final task evaluations. Hash seed 0, fixed original caps, no external API, no formal rerun. [Protocol](fresh_replication/protocol.json) · [Full results](fresh_replication/summary.json).

| Arm | Initial → final pass / 72 | Correctly completed | False complete |
|---|---:|---:|---:|
| 0.119.1 baseline | 67 → 72 | 71 | 0 |
| 0.120 | 66 → 71 | 69 | 0 |
| Empty-start cumulative | 62 → 72 | 72 | 0 |

The final version has 0 lower errors, 1 higher error and 71 ties versus baseline. Root07 range change remains unresolved (MSE 0.000112055); root09 initial apparatus and root12 range pass but remain unresolved. Thus formal accuracy did not reproduce [candidate v1 development](development_v1/summary.json), where known chains improved 8/12 → 12/12. V1 excludes the final latest-trial policy; final 0.120 additionally gives confirmation priority only to the latest confirmed trial. Frozen snapshots remain separate. `development/protocol.json` is a prepared, superseded plan that was never run; the executed candidate is `development_v1`.

[Costs by source, learning and terminal](fresh_replication/cost_totals.json): successor learning computation falls 9,755,291 → 9,196,922, but source-inclusive total computation is old/new/empty **12,564,760 / 12,006,391 / 9,490,027**. New versus old saves 4.44% total computation while total measurement rises 211,872 → 217,836 (2.81%). New total computation is 26.52% above empty-start. Final-model deployment measurement across the fixed terminal workload is 3,072 / 9,216 / 3,072 in table order. Research CPU including sources is 787.80 / 731.37 / 550.31 seconds; it excludes checkpoint I/O and terminal scoring. Each seeded arm counts the same genuine 12-source learning cost once; empty-start has no source cost and then learns cumulatively.

[Next gap](fresh_replication/next_gap.json): judge method applicability and conditionally retain, simplify or withdraw old mechanisms, measuring source-inclusive cross-task return. In root07 the new route spends 32,229 computation on program revision before structural research; it exhausts computation with 139 observations still affordable and no relation experiment. Old and empty each buy 13 useful relation experiments and finish. Stop small ranking adjustments to this one case; the complete result does not establish stable net benefit over local discovery. No process remains running and no additional experiment is pending.
