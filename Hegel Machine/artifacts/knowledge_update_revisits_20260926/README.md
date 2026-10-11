# Knowledge updates and unresolved-task revisits

Completed offline study: 12 new 9986-series apparatus, six tasks per session, four arms; `PYTHONHASHSEED=0`, no external model API. There were **216 actual first visits + 34 actual revisits**. The deletion arm reuses the new arm's 72 first-visit checkpoints and charged costs; all four arms contain 288 task records, with initial/final terminal scoring only after all learning completed. [Protocol](fresh_replication/protocol.json) · [Runner](../../scripts/run_knowledge_update_revisits.py).

Each real session preserves paid banks and continuation state. Initial caps alternate 256/384 observations, 180k/240k computation and 3072/4608 measurement; revisits and screening use only the shared unspent pool. Twelve genuine learned sources count once per seeded comparison arm. Empty-start begins empty and then accumulates knowledge normally. Matched deletion copies only revisit positions/allowances, restores each task's own post-initial library and passes no successor candidate, program, planned probe or prediction.

[Frozen 0.119 formal results](fresh_replication/summary.json):

| Arm | Initial → final pass / 72 | Correctly completed | False completion | Revisits |
|---|---:|---:|---:|---:|
| 0.118 baseline | 62 → 63 | 56 | 0 | 1 |
| 0.119 knowledge revisit | 62 → 64 | 59 | 1 | 10 |
| Matched knowledge deletion | 62 → 64 | 57 | 0 | 10 |
| Empty-start cumulative | 64 → 72 | 67 | 0 | 13 |

The new arm performs nine successor distinguishing experiments (33 paid rows), but formal prediction accuracy ties knowledge deletion. Learning computation is 10,022,331 / 11,941,912 / 13,870,333 / 11,524,921 in table order: new research costs 19.15% more than baseline, although it costs less than matched deletion. Full source, screening, revisit, terminal, deployment and CPU accounting remains in the summary. [Matched cases](fresh_replication/matched_revisit_cases.json) preserve successes and failures. Development's 14/15/14/18 result remains separately recorded in [development](development/summary.json); its independent accuracy advantage did not replicate formally. The formal runtime only adds the declared early skip of certainly inapplicable screening to that development version.

The root02 false completion was a real continuation bug: after research narrowed a new candidate's range, tied confirmation restored an old answer outside that range while retaining the new closure state. [Paid-evidence diagnosis](fresh_replication/root02_paid_diagnosis.json) separates this from a merely unseen terminal boundary. The **0.119.1 known-root02 replay** restores its original first-four-visit state, repeats the same last two tasks and natural revisit within unchanged caps, and reaches [6/6 correctly completed](repair_replay_root02/summary.json): affected-task MSE 2.12e−5 → 7.83e−20, unchanged observations/measurements, +858 chain computation. This is repair validation; the original formal result remains 64/72 with one false completion. A single-arm dispatcher error after successful replay learning was fixed without rerunning research; [runner note](repair_replay_root02/runner_note.json) records it.

[Next gap](fresh_replication/next_gap.json): inherited transfer paths consume computation while more useful response discovery and distinguishing experiments run out of budget. Formal root03/root06 confirm this pattern; empty-start solves all 72 tasks. Prioritize research value relative to the current best explanation and selective withdrawal of obsolete experience. The final production source additionally expands called programs in the refuted-incumbent comparison helper; this does not change root02's plain-body path and is not substituted into frozen snapshots. No formal seed, budget, result or runtime was changed, and no process remains running.
