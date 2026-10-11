# Development continuation and pending method intervention

The learner now receives current-source validation coverage after each write and
practice action. Publication criteria are unchanged. This run starts from the last
actual candidate of the [failed online run](../temporal_paired_online_repaired_luna_20261006/RUN_STATUS.md),
with original predictive knowledge and empty practice observations/trials.

Case_a confirms a revised model at MSE 1.90e-18. The learner then edits its program,
and case_b confirms the unchanged prior at 1.06e-17. Publication correctly reports
that the current version still lacks case_a confirmation. Call 6 fails with
`Selected model is at capacity`; the provider reports `timeout: false`.
**No method is published and no transfer comparison has run.**

This continuation uses 240 observations, 432 measurement units and 22,449 charged
computation units. Five successful API responses report 194,933 tokens, with
65,792/191,777 input tokens cached; the failed response supplies no token usage.
Together with the preceding attempt, development uses 23 calls, at least 1,217,417
reported tokens, 936 observations, 1,512 measurement units and 89,988 computation
units. The exact token total remains unknown. Separate source versions' successes
are not combined into a confirmed procedure.

The retention/deletion runner is ready: both live agents share the same procedure
interface, original predictive files, final verbal lesson and resource limits;
only the published procedure and its entry differ. It records explicitly supplied
failed learning costs once per condition. Fourteen focused tests cover the runtime,
publication state and comparison runner. These are interface tests, not LLM results.

## Next step

When Luna is available, complete paired validation and publication of one unchanged
candidate, then run the prepared four-pair method intervention. Keep both development
attempts in its learning costs. The present feedback change has not established an
independent learning benefit, and the comparison has no performance result yet.

Evidence: [result](learning/result.json), [protocol](protocol.json),
[unpublished source](learning/shared/consolidated_method.py).
