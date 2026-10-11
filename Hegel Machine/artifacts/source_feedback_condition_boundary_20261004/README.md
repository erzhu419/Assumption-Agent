# Source feedback condition boundary

This is a small offline boundary study for the existing real
`failed_successor` and `source_competition` fixtures.  It generates five paid
prefixes from `ConflictLagEnvironment`, then runs the real bad and repaired
source continuations with the same research limits.  The held conditions are
not synthetic scores: the rows and source returns come from the fixture
environment and ordinary `UnifiedResearchAgent` execution.

The five prefixes expose an important limitation.  The bad source is rejected
only for the narrow `single_query00` condition (`paired_gain < 0`).  In the
other four conditions it can receive a positive paired return, including cases
where the final status remains unresolved.  Therefore source feedback is
condition-specific and the single fixture does not establish universal bad
source withdrawal.  The receipt now records `confirmation_status`: the
positive unresolved `initial` bad-source rows are `provisional`, while the
positive completed rows are `confirmed` and the negative rows are `rejected`.
Only the confirmed status is allowed to create positive scheduler credit.

The five seeds are repeated on purpose to expose search variation.  The
laboratory transitions and paid rows are deterministic, while the seeded
candidate/refit search can produce different paired returns.  These are
multi-seed search runs over one fixed task family, not independent task data
or a long-run net-benefit claim.

Run offline with:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_condition_boundary.py
```

The exact 50-row receipt is in `results.json`.  External model calls are zero.
