# Repeated source-return confirmation

This receipt exercises the new status transition for two distinct paid
response draws with the same coarse condition and a nonzero leave-one-out
residual.  The first completed visit is recorded as `provisional`, with zero
scheduler credit.  A second completed visit under the same condition promotes
both records to `confirmed` with `confirmation_replicates=2`; only then does
`return_priority` provide positive reuse credit.

The two visits are a controlled status receipt, not a long-horizon performance
claim.  The full multi-task budget comparison is still pending.

Reproduce offline with:

```bash
PYTHONPATH=src python3 scripts/run_source_feedback_confirmation.py
```

External model calls are zero.
