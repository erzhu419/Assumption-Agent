# Aggregate-response process

All 56 purchased observations match target = total measured quantity + total measured impulse (maximum error 3.4e-16). Position, focus changes, repeated inputs, and trajectory resets do not alter this relation in the acquired evidence. The coarse channel preserves the required totals and costs less than fine measurements. Raw quantity commands are not the same as measured quantities.

`measurement_totals.py` extracts additive quantity and impulse totals from either supported measured view. It is not a predictor for arbitrary future mechanisms: refit response coefficients and test dependence on object relations and history using each task's observations. Group means cannot replace additive totals.
