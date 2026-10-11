# Source feedback selection study (completed)

This is the frozen offline comparison after 0.180.0. The baseline is a copy
of the active 0.180.0 source made before the source-feedback edit; the
candidate arm imports the live source after that edit. Both arms use the same
paid prefixes and the same budgets. The study completed for both cases.

This directory is predecessor evidence for the 0.181.0 continuation. Its
saved `research_return.condition` records predate the response-signature gate,
so they remain valid for the original 0.180→early-0.181 comparison but are not
current signed-return records. A new paid visit is required before those
returns can receive current scheduler credit.

The old ordinary-source study recorded four actual positive returns but found
zero later consumers. This study keeps the two natural ordinary-source cases
and records whether a later screen sees the return and changes its queue. It
also preserves the existing real negative-return fixture: a rejected binding
must reorder a condition-matched source/replay queue while leaving the source
available for another condition.

No return, gain, terminal score, or revocation is injected. The candidate
created two zero-cost feedback refresh events after real returns in the
three-source case; the final visit choices, resource bills and terminal scores
were unchanged. The single-source case had no later competing offer. A
separate small receipt in `../source_feedback_action_divergence_20261004/`
covers a real positive/negative queue and its matched no-return action
divergence. The later same-fixture boundary receipt in
`../source_feedback_condition_boundary_20261004/` records whether each return
was `confirmed`, `provisional`, or `rejected`; a positive paired gain from an
unresolved visit stays provisional and contributes no positive source priority.

Run only after the source-feedback implementation is frozen:

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_cognition_unified_source_returns.py
PYTHONPATH=src python3 artifacts/source_feedback_selection_20261004/run_study.py \
  --phase prepare
PYTHONPATH=src python3 artifacts/source_feedback_selection_20261004/run_study.py \
  --phase learn
PYTHONPATH=src python3 artifacts/source_feedback_selection_20261004/run_study.py \
  --phase evaluate
```

The protocol and baseline snapshot are the only frozen inputs at this stage.
No external model or network call is involved.
