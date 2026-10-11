# Parameterized task-family source feedback study

This receipt extends the conflict fixture with a real parameterized transition
law.  Each task changes the observed target relation
`y_t = a*x_t + b*x_{t-2} + noise`; the input probes, paid prefix, source
concepts, and continuation limits remain visible and are executed by the
ordinary unified research loop.  The two arms use identical screened offers
and budgets.  The matched arm removes only prior `research_return` records.

There are three variants, two seeds, and two arms (12 rows): the original law,
a changed current-input coefficient, and a noisy version of the original law.
On the original law, feedback selects `concept_0002` and reaches held-out loss
near zero, while the no-return arm selects `concept_0001` and remains
unresolved.  The changed coefficient law first exposed stale transfer: before
the response-signature fix, feedback withdrew `concept_0001` and selected the
wrong `concept_0002` (held-out loss `8.31e-4`).  With the fix, the measured
response regime no longer matches that old return; both arms select
`concept_0001` and reach held-out loss below `1e-10`.

This is a guarded result, not a cross-task performance claim.  The new
condition records a coarse paid-response signature and a single-fit response
residual.  The residual is deliberately computed once per condition; the
earlier leave-one-out implementation made repeated screening unnecessarily
slow without changing this routing decision.  The signature blocks the
observed coefficient-shift failure and keeps the noisy variant from importing
the old negative withdrawal without adding another measurement: both noisy
arms select `concept_0001`, remain unresolved, and emit no feedback event.
This is a routing guard, not a confirmation of the noisy mechanism; repeated
evidence is still needed before that source can earn positive credit.
Contribution attribution and stable long-run net benefit remain unproven.

Reproduce offline with:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_task_family.py
```

The exact 12-row result is in `results.json`; external model calls are zero.
