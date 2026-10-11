# Shared-budget multi-task source feedback

This receipt runs six tasks in one `UnifiedResearchSession`: two base-law
tasks, two shifted-coefficient tasks, and two noisy tasks.  Both arms use the
same task order, per-task limits, screened offers and total resource budget.
The feedback arm retains source returns and refreshes queued offers; the
matched arm removes source-return records after each visit.

The feedback arm selects the repaired source on both base-law tasks and the
original source on shifted and noisy tasks.  The matched arm selects the
original source on all six tasks.  Both arms consume the same total resources:
246 observations, 0 measurement cost and 40,434 compute.  Base-law held-out
loss is near zero with feedback and `8.31e-4` without it.  The shifted law is
near zero in both arms.  The noisy tasks remain unresolved; the feedback arm
does not convert the noisy response into positive source credit.

This is the first shared-budget long sequence for the new response gate.  It
shows an action and quality difference on the repeated base law while keeping
the coefficient-shift and noisy cases from receiving stale negative transfer.
It is still a controlled fixture, not evidence of stable general long-run
net benefit or a complete causal retain/delete study.

Reproduce offline with:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_long_horizon.py
```

External model calls are zero.
