# Joint response discovery experiment

Baseline source: `baseline/src` (0.110.0). Development uses the previously seen successor07 at 192/384 rows in warm and empty modes. Fresh replication uses 12 new instrument seeds 99221001–99221012, name seeds 99222001–99222012, wiring seeds 99223001–99223012, and task seeds 99224001–99224012; budgets 96/192/384; warm and empty modes; two code arms (144 runs).

`replication/` was stopped because its instrument seeds overlap the previous study; preserve it as development-only evidence and exclude it from fresh replication. Use `fresh_replication/` for new-task comparisons.

Development revisions are preserved separately: `development/` exposed lost-incumbent and local-range failures; `development2/` retained the better explanation; `development3/` crossed the range-search local minimum; `development4/` also retained the simpler response families. Formal `fresh_replication/runtime/src` is frozen at 0.111.0; all 144 learning and 144 terminal evaluations completed. See `fresh_replication/aggregate_diagnostics.json` for actual CPU and metered costs, and `fresh_replication/empty384_failure_diagnosis.json` for the remaining boundary/confirmation failures.

Run from the project root, with the selected revision frozen into the protocol's revised runtime before learning:

```sh
PYTHONHASHSEED=0 PYTHONPATH=src /usr/bin/python3 scripts/run_relation_binding_budget_study.py --phase learn --output artifacts/joint_response_discovery_20260922/development --arm revised
PYTHONHASHSEED=0 PYTHONPATH=src /usr/bin/python3 scripts/run_relation_binding_budget_study.py --phase learn --output artifacts/joint_response_discovery_20260922/fresh_replication --arm revised
```

For each cohort, finish all learning in both arms before `--phase evaluate` for either arm, then `--phase summary`. The runner enforces this ordering. Four independent 64-row evaluation trajectories per task, with paired seeds across arms. Actual saved source knowledge is shared only by warm runs; every empty run begins with an empty library. This is a set of one-task successors, not independent source-training or long-term sequences.
