# Prior-first confirmation and repair

The frozen Luna-authored program passes fresh confirmation and terminal evaluation
on all four cases: two earlier calibrations, each with an unchanged and a changed
mechanism. Execution makes no LLM calls. The second calibration is unavailable
during program authoring; its supplied predictor role transfers without a code edit.

| Case | Mechanism | Initial confirmation MSE | Result | Observations |
|---|---|---:|---|---:|
| 00, second prior | Unchanged | 1.23e-16 | Retain original predictor | 48 |
| 01, first prior | Changed | 0.5322 | Measure, refit, confirm | 144 |
| 02, second prior | Changed | 2.3136 | Measure, refit, confirm | 144 |
| 03, first prior | Unchanged | 1.79e-18 | Retain original predictor | 48 |

Failed 48-row confirmations return actual counterexamples. The same program then
buys 48 measured observations, fits instrument and response hypotheses, and requests
another 48-row confirmation. Both repaired predictors pass. All four executions
stop before their separate 256-row terminal evaluations; maximum terminal MSE is
8.77e-16. Totals are **384 observations, 288 measurement units and 15,820 charged
computation units**. The earlier always-refitting program solves the same four cases
using 480, 864 and 93,828 respectively. Changed cases cost more observations here
(144 versus 120); unchanged cases cost less (48 versus 120).

The first paired learner completed in five calls and published that always-refitting
program. Subsequent reuse-and-revision learning exposed a host bug: initial model
submission incorrectly required paid training rows before fresh confirmation.
That run was interrupted. The repaired executor skips the absent training diagnostic
but still purchases and evaluates fresh confirmation. This replay uses the exact
last recorded Luna write (step 8), with no manual policy edit or trial-state carryover.
The two authoring runs record 15 completed calls and 665,445 tokens; an interrupted
in-flight call may add unrecorded usage. Replay itself uses zero tokens.

## Limitations and next step

This is a successful frozen-candidate replay after runtime repair; online learning
and publication of this candidate remain unfinished. Reuse/revision objectives,
prior roles and numerical fitting families are supplied. Predictive rejection does
not independently identify an instrument fault. These are two previously studied
calibrations, and the cost comparison changes acquisition patterns as well as reuse.
Complete prospective learning under the repaired runtime, then compare method
retention/deletion with identical predictive memory and verbal experience.

Evidence: [summary](summary.json), [protocol](protocol.json),
[unaltered candidate](candidate.py),
[earlier fixed-refit result](../temporal_paired_method_luna_20261006/summary.json),
[interrupted learning](../temporal_paired_reuse_method_luna_20261006/interruption.json).
