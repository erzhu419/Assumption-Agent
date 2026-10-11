# Composition comparison for the first paper

The [protocol](protocol.json) fixes 30 apparatus instances, four algorithms and
six tasks per sequence. [summary.json](summary.json) contains every completed
case, aggregate costs and paired apparatus-level bootstrap intervals. Per-case
JSON files preserve task losses, confirmation outcomes and actual program calls.
The development instance is separate from the registered cohort.

All 30 cases completed: full 180/180, task-local 179/180, passive 87/180 and
dense fixed basis 0/180. Full saves 59.02%/74.82%/76.25% of the task-local
observation/measurement/charged-computation totals. Aggregate training CPU is
24.17% lower, but its paired interval includes zero; the slowest full case is
retained. Seventy passive confirmations fail subsequent independent evaluation.

`run_cohort.py` imports only the adjacent frozen `runtime/`. The reusable entry
point is `scripts/run_composed_offline_comparison.py` in the repository. All
learning finishes before terminal evaluation. No external models are called.

The task-local arm retains within-task learning, starts a fresh library per task,
and cannot borrow other tasks' budgets. Passive experiments use four fixed
24-row trajectories and the legal fine view. The fixed basis is a dense fit of
quadratic and two-step-history terms. Equal ceilings do not imply equal spending.

The manuscript and research plan in `paper/` interpret the results. Generate the
figure with `python3 scripts/plot_composed_offline_comparison.py` from the repo.

## Known issue

The new four-arm integration test and three passive tests passed. One existing
passive reuse test still fails its expectation of nonempty `used_sources`; the
same failure was reproduced with the passive module from git HEAD. The legal-view
repair does not resolve that separate expectation. The new integration-test
receipt is `../composed_offline_comparison_20261005_tests.xml`.
