# Same-budget unseen continuation check

This artifact compares the real `concept_0001` (bad) and `concept_0002`
(repaired good) successor continuations from the existing
`failed_successor`/`source_competition` fixtures.

Both arms start from the same target paid prefix and an identical cloned
source-library payload. The repaired good concept was fitted from the
fixture's earlier real paid rows before the two arms are created; this is the
knowledge being compared, not a new observation during either arm. Each arm
uses `observation_budget=33`, `compute_budget=20000`, and actually spends
`observation_cost=24`, `compute_cost=120`. The 12-row later trajectory is held
out until after each research run. The good arm reaches heldout loss
`9.725566e-20`; the bad arm reaches `0.0027531977`, while their costs are
identical. This is a fixed-budget heldout check, not evidence of a stable
long-sequence net gain.

Reproduce offline with:

```bash
cd "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine"
PYTHONPATH=src:tests python3 scripts/run_source_feedback_unseen_eval.py
```

The command rewrites `results.json`; it makes zero external model calls.
