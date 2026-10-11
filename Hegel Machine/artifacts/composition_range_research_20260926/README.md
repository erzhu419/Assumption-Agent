# Composition range research, 2026-09-26

`baseline/src` is the current 0.113 tree frozen before changes. `development` preserves the first implementation and its two newly exposed false completions; `development_final` preserves the final 0.114 implementation and the same eight development pairs. Development baseline reports were reused verbatim only after direct source-tree comparison, with original runtime and timing metadata retained.

`fresh_replication` contains 144 learning reports and 144 independent terminal evaluations: 12 new successor apparatus, two response conditions, three budgets, and two arms. All learning finished before any terminal evaluation. The 12 actual 0.113 source libraries are reused from the prior experiment; these are not new source-learning episodes. Each paired successor starts independently from identical serialized source knowledge.

Use `scripts/run_boundary_method_transfer_study.py --phase {learn,evaluate,summary} --output artifacts/composition_range_research_20260926/fresh_replication --arm {baseline,revised}`. Summarization does not need `--arm`. Runtime source paths, all seeds, budgets and source-report paths are in each frozen `protocol.json`.

For range-change tasks with 384 observations, completed-and-terminal-pass improves from 0/12 to 12/12. Across all 72 successors per arm, terminal pass improves from 14 to 28; completed-and-pass from 14 to 25. False completed remains 3 in each arm, with different cases. Successor observations increase 11.41%, measurements 10.22%; metered compute decreases 1.11%, observed CPU 3.30%.

## Limitations

No 192-budget task passes terminal evaluation. Medium-budget calibration/remaining-range decisions still produce three false completions and three terminal-passing but unresolved cases in the revised arm. `fresh_replication/remaining_failure_diagnosis.json` records these after the frozen study; no terminal result was used to revise this runtime. All experiments are local and offline; no model API was called.
