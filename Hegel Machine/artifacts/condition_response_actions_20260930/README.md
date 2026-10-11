# Condition-response research action rewriting

Baseline is the completed 0.136.0 corrected formal runtime, frozen as 458 Python files before 0.137 work. All experiments are offline; external model calls are zero.

The new physical laboratory adds the shared continuous contribution `.4*g1-.3*g2` to both quadratic response branches. Four controls have unknown response/condition roles. The seven stages retain plain W3, current AND W3, historical OR W5, layered AND W7, renamed/scaled layered XOR W7, return AND W3, and plain W5. Public TaskSpec contains only control names, bounds and a neutral description.

[New development](development/protocol.json) uses apparatus 9915301–9915302. [Previous failure development](development_previous_failures/protocol.json) separately revisits the three unresolved old formal tasks through their full two-apparatus sequences; those results cannot be new formal evidence. Its unchanged 0.136 baseline learning is copied without terminal evaluation. [New formal](fresh_replication/protocol.json) uses twelve apparatus 9915401–9915412: 252 arm-tasks across old cumulative, new cumulative and new independent.

Initial allocations are 320 observations and 320000 computation per task, with 2240/2240000 cumulative chain limits and natural revisits funded only by unspent visited-task allocations. No measurement channel is part of this formal family. All learning/revisits must end before common four 64-row terminal trajectories; every negative and actual cost is retained. Independent is a knowledge-reset control, not vanilla Code or LLM.

[Runner](../../scripts/run_condition_response_action_study.py). The 0.137.0 candidate is frozen as 459 Python files in each development cohort. The [first development](development/summary.json) remains 13/14 cumulative and 14/14 independent; the cumulative return failed at MSE 0.244962 despite two response-curve rewrites. The [corrected development](development_corrected/summary.json) regressed to 12/14 cumulative, with historical OR and return unresolved. Both snapshots and all negatives are preserved. The [final development](development_final/summary.json) solved 14/14 in each new arm after retaining paid transfer support, purchasing the deficit to 48 training rows and separately acquiring 24 comparison rows. Its source was copied unchanged to formal runtime.

[Previous failure development summary](development_previous_failures/summary.json): baseline solved 11/14; both new arms solved 14/14. Each of the three earlier unresolved cases executed and adopted a joint cross experiment and boundary refinement. This remains development evidence.

[Formal summary](fresh_replication/summary.json): all learning and natural revisits finished before all 36 common terminal jobs. Each arm has 84 tasks. Correct completions equal solved tasks in this table; false completions are reported separately. Costs include the complete learning sequence and revisits, excluding terminal scoring.

| Arm | Solved | False completed | Mean terminal MSE | Observations | Computation | Research CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old cumulative 0.136 | 26/84 | 2 | 7.92237 | 23313 | 23783890 | 2825.17 |
| New cumulative 0.137 | 84/84 | 0 | 4.99e-12 | 23421 | 8718226 | 1167.35 |
| New independent 0.137 | 83/84 | 0 | 1.78e-5 | 23904 | 7231524 | 882.44 |

For the 60 conditional tasks including return, solved counts are 2/60, 60/60 and 59/60. Return alone is 0/12, 12/12 and 12/12. All twelve cumulative apparatus chains actually rewrote both condition-response experiments; 54 conditional tasks acquired both curves, while six successful return tasks reused prior explanations (maximum terminal MSE 2.64e-12). All 60 independent conditional tasks acquired both curves. The single independent failure, root07 historical OR W5, remains unresolved at MSE 0.001492374 despite executing both curve experiments, the joint cross experiment and boundary refinement.

## Limitations

Cumulative learning uses 2.02% fewer observations than independent learning, but 20.56% more computation and 32.29% more CPU. Stable net benefit from accumulated knowledge remains unestablished. These are noiseless finite quadratic/Boolean laboratories, not vanilla Code or LLM comparisons. Paid condition instruments are separate focused development evidence (terminal MSE 0.01308) and are absent from this formal family. Formal failures are retained without subsequent source changes or parameter adjustment.
