# Adaptive boundary experiments

Baseline source frozen at 0.111.0 in `baseline/src`. Development reuses six previously observed reports and terminal evaluations verbatim; `development/baseline_reuse.json` records original locations. Reused runtime and timing fields are unchanged and are not fresh reruns.

Fresh replication: 12 new apparatus/name/wiring/task seeds 99321001–012 / 99322001–012 / 99323001–012 / 99324001–012; warm and empty libraries; budgets 96/192/384; baseline and revised code (144 learning jobs). All learning ends before any terminal evaluation. Terminal uses four paired independent 64-row trajectories. Runs are fully offline.

Frozen protocols distinguish controller completion, terminal MSE ≤ 1e-7, their intersection, and false completion; summaries retain observations, measurement cost, metered compute, actual CPU, and interval revisions without reduction.

`development/` freezes the initial route through complete-response interval experiments. `development2/` additionally preserves compute for assimilating the next purchased relation observations and confirmation; both versions retain the six original development cases. Four previously failing empty-384 cases pass terminal evaluation in both versions, and the warm-384 / empty-192 regression cases are unchanged. All reused baseline CPU times remain from their original executions.

Formal `fresh_replication/runtime/src` freezes 0.112.0. All 144 learning jobs and all 144 terminal evaluations completed. Empty-384 completion/terminal/intersection changed 11/8/8 to 12/12/12, with false completion 3→0; warm-384 remained 12/12/12. Lower-budget terminal results are unchanged, including one false completion in empty-192. Actual aggregate CPU rose 228.58→258.79 seconds (+13.22%); empty-384 CPU fell 14.51%, observation cost rose 2.97%, and measurement / metered compute fell 3.72% / 3.10%. `aggregate_diagnostics.json` keeps all groups and cases.

Interval diagnostics count the union of admissible cells per parameter, because competing candidate records may repeat the same parameter. Excluding each research chain's initialization, empty-384 repeated nonshrinking conditionings changed 4→0; legacy blind boundary revisions without reduction changed 91→0. Read-only failure analysis is in `fresh_replication/remaining_failure_diagnosis.json`; no production tuning followed formal terminal results.
