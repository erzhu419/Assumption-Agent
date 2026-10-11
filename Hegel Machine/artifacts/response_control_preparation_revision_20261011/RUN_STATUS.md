# Prepared controls and finite refit — 2026-10-11

Completed 24/24 executions on the existing 12 fixed conditions. Same baseline, nine initial declarations, available action set and budget limits; model calls/tokens and unsupported target confirmations are zero. The first candidate's four executions remain separate in `response_control_preparation_20261011`.

| Group | Baseline task / C5 confirmation | Candidate task / C5 confirmation |
|---|---|---|
| Cubic, 8 conditions | 6/8 / 6/8 | 7/8 / 6/8 |
| Square, 2 controls | 2/2 / 2/2 | 2/2 / 2/2 |
| Aliased, 2 controls | 0/2 / 0/2 | 0/2 / 0/2 |

The other 10 pairs have identical outcomes, complete action traces, action counts and research-compute accounting. Total actions: 1582→1614; compute: 8893388→9487209. CPU and wall-clock timing are execution-dependent.

`original1`: paid C1=16 activation (observation 130) reduced conditional target support 224→200 and triggered one finite joint refit. Its new settings 4687 and 4876 were both tested and refuted; the task stopped with 194 settings remaining.

`shifted1`: paid C3=61 activation (observation 145) reduced support 20→18 and triggered one finite joint refit. The agent selected 3650, then encountered actual activation at 3662 while moving there (observation 148), completing the task. The requested 3650 prediction was not confirmed.

Evidence: `group_summary.json`, `diagnostic_revisions.json`, `fixed_program_diagnosis.json`, and each condition's research/action traces. Forty targeted tests passed (`validation.json`). Remaining work is predictive selection and supported-function reuse; this batch establishes an executed research continuation and one additional task completion, not an additional confirmed target prediction or a new held-out result.
