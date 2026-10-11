# Live research choice with historical methods

**All four Luna agents choose a distinguishing experiment before fitting another
response. All four confirm and pass terminal evaluation with zero MSE.**
The existing v0.93 action policy now diagnoses paid candidate agreement and
returns an executable plan from the existing response/instrument planners.

The same two window/event pairs, paid calibration, candidate laws and limits as
the [earlier pilot](../temporal_historical_methods_luna_20261006/RUN_STATUS.md)
are reused. Only the measured historical method book differs within each pair.

| Target totals | Retained | Removed |
|---|---:|---:|
| Confirmed and terminal-correct | 2/2 | 2/2 |
| Observation rows | 248 | 248 |
| Measurement units | 456 | 456 |
| Charged computation | 26,329 | 31,042 |
| LLM calls | 6 | 6 |
| Reported tokens | 206,813 | 179,812 |

Each run uses `research_choice → query/assess_response → finish`, with four new
discriminating rows. Both retained runs execute and support a historical method;
both removed runs learn a new method. No run performs the earlier redundant
zero-response fit. Retention saves 15.2% computation and uses 15.0% more tokens.
Input cache hits are 139,264/384,784 (36.2%); all 12 calls use subscription Luna.
[Raw results](summary.json), [protocol](protocol.json), and per-episode training
traces preserve the actual selections and charged preparation costs.

## Limitations

These are two development pairs with supplied candidate laws. The larger old
researcher is available but none of these four agents invokes `legacy_research`;
its full-library execution and transfer are covered by offline integration tests.
Historical acquisition cost remains recorded separately. This is local method-use
evidence, not a general long-run advantage or a rerun of all181 historical studies.

Next: use the existing whole-session entry to join these successful research
episodes with pending-task return and source-attributed feedback.
