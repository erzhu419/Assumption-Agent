# Control preparation diagnostic — 2026-10-11

Completed 4/24 planned executions: original1 and shifted1, paired baseline/candidate. The remaining 20 were not run for this first candidate.

Both candidates executed and recovered an activated reference on the new grid. Conditional target support shrank 224→200 and 20→18, respectively; neither target task or requested C5 confirmation succeeded. All four executions stopped naturally; unsupported target confirmations, model calls and tokens were zero.

Actions: original1 123→132; shifted1 138→147. Research compute: 387870→550205 and 817104→960912. Results are retained separately from `response_control_preparation_revision_20261011`; they must not be pooled across candidate versions.

Next revision reuses the same baseline, cases and budgets and permits a finite joint-candidate refit only after actual reference evidence shrinks support.
