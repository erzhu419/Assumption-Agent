# Boundary method transfer

`fresh_replication/protocol.json` fixes 12 new independent source apparatus and 216 successor runs: two response conditions, three budgets, three arms. Each source is actually learned from an empty library using frozen 0.113.0. Every successor restarts from its root's identical source library; `without_methods` removes only the inherited `program_experiment_methods` records, and can learn new methods locally. This is paired transfer, not a multi-generation sequence.

The baseline uses frozen 0.112.0. All source and successor learning finishes before any successor terminal evaluation (four 64-row trajectories per case). Source learning costs are reported separately and included once in each arm's full-cost comparison. Full research traces and resulting methods are retained; successor libraries that are never reused are not duplicated.

Run from the project root with `PYTHONHASHSEED=0` and `/usr/bin/python3 scripts/run_boundary_method_transfer_study.py --output artifacts/boundary_method_transfer_20260922/fresh_replication`:

- `--phase sources`
- `--phase learn --arm baseline`, then `revised`, then `without_methods`
- After every learning run exists: `--phase evaluate` for each arm, then `--phase summary`

Development is separate: `development` retains 16 baseline runs with previous real source libraries; `development_fresh_sources` retains two newly trained sources and 12 baseline successor runs; `development_rebound` reuses those exact sources/baseline reports and completes all 36 paired runs and terminal evaluations. `final_failure_check` repeats only the two known 256-row calibration failures using the final runtime. None uses formal terminal results to select the implementation.

Completed: 12 source runs, 216 successor learning runs, 216 terminal evaluations. `fresh_replication/summary.json` includes all paired results and costs. Revised background rewriting is measured successfully in 8 successor runs from 6 independent roots. Terminal accuracy matches 0.112 throughout; relative to removing inherited methods, one additional 256-row response-change task is solved, and that task did not use the new background rewrite. Every same-response radius-change task remains unresolved across all budgets. These results establish execution of the new transfer mechanism, not overall accuracy improvement over 0.112 or a long-run net benefit.
