# Autonomous confirmation after the shared submission repair

Both conditions pass fresh confirmation and terminal evaluation on 4/4 tasks,
with zero false confirmations. Eight fresh agents reuse the original pilot's
cases/seeds, published method, predictive memory, lesson, Luna model and limits.
Both conditions receive the repaired direct-finish interface. All training ends
before terminal scoring; the original 3/4-versus-3/4 pilot remains separate.

| Transfer measure | Method retained | Method deleted |
|---|---:|---:|
| Confirmed and correct | 4/4 | 4/4 |
| LLM calls | 18 | 16 |
| Reported tokens | 794,795 | 848,375 |
| Observations | 408 | 372 |
| Measurement units | 648 | 396 |
| Computation units | 30,224 | 25,028 |
| Python seconds | 2.78 | 6.40 |

All four retained agents submit through the unchanged learned program: ten actual
invocations, with original-predictor reuse in both stable cases. All four deleted
agents directly submit refitted predictors; the repaired interface registers their
selected paths before paid confirmation. Neither condition writes manifests or
encounters submission/provider/decoding failures. One retained response fit lacks
a separate validation trajectory; the LLM obtains another and retries successfully.
The previously blocked retained case now autonomously passes fresh/terminal MSE
1.23e-19/1.69e-19 in six calls. Input cache hits are 697600/1633331 (42.7%).

## Limitations and next step

Retention saves 6.3% of reported tokens but increases calls, data and charged
computation; no overall advantage is established. Shared learning costs remain
included once per condition, with their exact token total unknown. Same-case
stochastic reruns do not isolate repair causality or establish unseen-task
performance. The narrow autonomous completion loop is now observed; next freeze
the program and test applicability under new calibrations and probe trajectories.

[Full results](summary.json) · [Protocol](protocol.json) ·
[Original pilot](../temporal_paired_retention_luna_20261006/RUN_STATUS.md)
