# Temporal experiments and successor-method opportunities

The protocols in `development/protocol.json` and `fresh_replication/protocol.json` were saved before this round's learning began. Both use the existing observed-condition laboratory and runner, search proposals only, and no external model calls.

- Baseline: retained 0.140.0 runtime from `partial_method_choice_20260930/fresh_replication/runtime/src`, with 465 Python files.
- Development: known prior negative apparatus 9915805 and 9915807. Baseline learning checkpoints are reused; prior terminal evaluations are not copied. This cohort is now development only.
- Formal: 12 fresh apparatus, 9915901–9915912, seven stages, three arms, 252 tasks. Both cumulative arms start empty; independent resets the full library and unused allocation at each task.
- Original objective and budgets are unchanged: MSE ≤ 1e-6, 320 observation rows, 320 measurement units and 320,000 computation per initial visit; cumulative revisits may spend only the shared unused allocations of already visited tasks.
- New terminal draws were fixed in advance: offset 910303, four trajectories of 64 rows. Terminal scoring starts only after every arm completes learning and natural revisits. Terminal evidence does not change the source or protocol.

Candidate source will be frozen after implementation and direct consumer tests, before any candidate learning. Full prior results and source snapshots remain at their original paths.

Completed: all learning and natural revisits in both cohorts finished before any terminal scoring. Development evaluated 42 tasks and formal evaluated 252 tasks, with no external model calls or source/protocol retuning. All frozen source copies and negative outcomes remain available.

Use `scripts/run_observed_condition_decoding_study.py` for learning/evaluation/summary; use `analyze_temporal_opportunities.py` for measured steps and method opportunities. The prior `partial_method_choice_20260930/analyze_partial_knowledge.py` can be run with either new cohort path for common knowledge lineage and whole-chain comparisons.

Candidate 0.141.0 was frozen after source and consumer tests, before candidate learning. The formal runtime is copied from the development frozen runtime. Both source copies contain 466 Python files. See each cohort's `candidate_source.json`.

Formal strict solutions rose from 60/84 to 63/84 (independent 48/84), but mean response MSE increased 12.26% and learning compute increased 54.47%. Method opportunities executed 14 real experiments and 14 revisits; 7 revisited tasks passed the final terminal criterion. This cohort does not establish stable net benefit.

Use `compact_results.json` for endpoint, cost and chain summaries. `fresh_replication/read_only_regression_diagnosis.json` evaluates frozen first-visit and final models on the same already fixed terminal trajectories; it shows the largest regressions arose on the first visit, and expensive revisits improved them only partly. No diagnostic evidence changed learning or the protocol.
