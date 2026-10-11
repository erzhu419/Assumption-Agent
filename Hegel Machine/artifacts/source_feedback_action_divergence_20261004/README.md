# Real mixed-source action divergence

This receipt is a small offline ablation built from the existing
`source_competition` fixture. The bad source has a real rejected continuation;
the good source is a real refit whose fresh continuation records a positive
return. Both arms receive the same two already screened successor offers,
paid prefix, and resource pool.

The candidate keeps the measured returns. It withdraws the bad offer and
selects the good source. The matched arm removes only the `research_return`
records, keeps both concepts and both offers, and selects the stale bad source
(with the good offer as a companion). Refreshing costs zero compute in both
arms and opens no laboratory. The receipt contains no injected gain or
revocation.

Run:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_action_divergence.py --phase all
```

This is the causal scheduler/action check. The next same-budget execution is
recorded in `../source_feedback_same_budget_20261004/`; it executes the two
divergent choices and compares later-task loss at identical visit cost. Neither
receipt establishes a multi-task long-run net benefit by itself.
