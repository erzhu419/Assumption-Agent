# Retention after a measured failure

The [protocol](protocol.json) specifies 30 new apparatus instances and three
interventions after the same learned A/B/C prefix and paid challenge of C.
The arms conditionally retain C, retain it without that challenge's feedback,
or delete C and its descendants. A and B remain available.

[summary.json](summary.json) reports every completed case, activation counts,
complete costs, the three successor phases and paired bootstrap intervals.
Each case also records the challenged program, source identity, initial feedback
state, actual removed IDs, terminal outcomes and executable source calls.
The development instance is separate from the 30-instance cohort.

`run_cohort.py` imports the adjacent frozen `runtime/`. The reusable runner is
`scripts/run_scoped_retention_comparison.py`. All arms complete learning before
terminal scoring. The common challenge is charged and reduces the first changed
task's remaining allowance. Suffix sessions share only their own library, with no
cross-task budget borrowing or cached session reports. No external models run.

Two targeted tests passed in `tests/test_scoped_retention_comparison.py`, including
an actual three-arm learning sequence and nested-source deletion. Regenerate the
figure using `python3 scripts/plot_scoped_retention_comparison.py` from the repo.

## Interpretation

All 30 registered cases completed and activated the intervention. Conditional and
unmarked retention each solved 180/180 tasks; global discard solved 159/180.
Retention reduced whole-sequence observations by 20.00%, measurements by 39.04%,
and charged computation by 26.59%. All arms solved all 30 original-condition returns;
retention reduced return computation by 77.66%. The two retention arms had exactly
equal errors and charged costs in every instance. This extra conditional record
had no independent measured benefit. Training CPU differences remain uncertain.

Discard failed 20 first-changed tasks and one changed return. Four first-changed
failures passed internal confirmation; 16 remained unresolved. Every failed
first-changed task had substantial observation, measurement and computation
allowances left. The next experiment varies which component changes and records
the actual research decision and stopping cause; the compact results here do not
identify one common failure cause.

Conditional feedback follows the existing ordinary-transfer record update.
It does not create a learned general scope predicate. The separate permanent
target-binding withdrawal API uses the same public identity for original and
changed object episodes, so it can also suppress a response when the old regime
returns; that is a distinct applicability limitation and is not the intervention
tested here. Challenge outcomes provide feedback, not a free extra fitting set.
