# Multi-condition source-feedback replication

This is a small offline extension of the real `source_competition` fixture.
It runs two bound conditions, two task IDs and three declared seeds, with the
same screened offers and continuation budget in both arms: 12 trials per arm.
The feedback arm applies the measured return before choosing; the matched arm
removes only return records and follows the old queue-first choice.

In the matching condition, feedback withdrew the bad source and selected the
repaired source in all 6 trials; all 6 completed with later loss near
`9.73e-20`. In the narrow condition the stored return did not match the new
condition, so the bad source remained and was selected in all 6 trials. The
matched arm selected the bad source in all 12 trials and resolved none. Every
trial used 24 observations, 0 measurement cost and 120 continuation compute.

The three seeds are stable repetitions of this deterministic fixture, not
independent stochastic environments. The result confirms condition-scoped
withdrawal and exposes the boundary where feedback must not be transferred;
it does not establish general multi-task or long-run net benefit.

Run:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_multitask.py
```

The run makes zero external model calls.
