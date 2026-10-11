# Multiple conditions and temporal-layer composition

Baseline: frozen 0.135.0, 456 Python files. Candidate: 0.136.0, 458 Python files. The corrected formal cohort adds one main-entry local-variable binding fix to the completed `development_repaired` runtime.

All three arms receive initial allocations of 320 observations and 320000 computation per task. Cumulative arms can reuse unspent visited-task allocations within the 2240/2240000 chain limits. The seven-stage sequence compares unconditional windows, current AND, historical OR, layered AND, renamed/scaled layered XOR, return to AND, and an unconditional control. Gate and response roles vary across apparatus.

Development uses 9915101–9915102; the formal roster uses 9915201–9915212 (252 arm-tasks). Learning and natural revisits finish in all arms before common four 64-row terminal trajectories. Both cumulative arms begin with empty knowledge; independent is an algorithm/knowledge control, not vanilla Code or LLM. All experiments run offline without external model calls.

Development results: baseline solved 4/14, repaired cumulative 13/14, repaired independent 14/14; no false completions. The cumulative return-task failure remains unresolved (MSE 0.13591), and cumulative research CPU was 5.50 times the independent arm. These remain negative evidence. The initial interrupted cohort and the corrected/final 13/14 outcomes are preserved in their own directories.

Protocols: [development](development_repaired/protocol.json), [corrected formal](fresh_replication_corrected/protocol.json). Evidence: [development summary](development_repaired/summary.json), [formal freeze](fresh_replication_corrected/runtime/candidate_freeze.json). Runtime source: [baseline](baseline/src), [candidate](fresh_replication_corrected/runtime/src), [study runner](../../scripts/run_composed_condition_study.py).

The first formal candidate run stopped on two cumulative roots: an unbound local `spec` in the main research entry point caused `UnboundLocalError`. The frozen runtime, failed checkpoints and [learning log](fresh_replication/new_learning.log) are retained; its last interrupted visits have incomplete paid-cost records. No common formal terminal evaluation ran. `fresh_replication_corrected` reran both new arms from empty knowledge with the binding fix and identical protocol; baseline learning was copied unchanged.

All 252 corrected formal arm-tasks and natural revisits finished before common terminal scoring. [Final summary](fresh_replication_corrected/summary.json):

| Arm | Solved | False completions | Mean MSE | Learning observations | Learning compute | CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 0.135 cumulative | 24/84 | 0 | 4.07279 | 23777 | 26267917 | 2217.77 |
| New cumulative | 81/84 | 0 | 0.228637 | 22307 | 16657422 | 3314.71 |
| New independent | 84/84 | 0 | 1.13e-16 | 21264 | 11961281 | 864.19 |

The new cumulative arm won/tied/lost 58/24/2 tasks against the old cumulative arm, with 94.39% lower mean MSE. Against the new independent arm it tied 81 and lost 3, using 3.84 times CPU, 4.91% more observations and 39.26% more computation. Three unresolved cumulative failures remain: root03 renamed/scaled XOR (MSE 6.46533), root05 historical OR (2.45524), and root05 layered AND (10.28493). These formal outcomes did not change source or protocol. The study supports measured multi-condition/time-layer discovery; stable benefit from accumulated knowledge and vanilla Code/LLM superiority remain unestablished.
