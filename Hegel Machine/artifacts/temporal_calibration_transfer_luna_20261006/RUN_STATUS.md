# Independent calibration transfer

The published program and original donor models remain unchanged. An independent
response seed separates changes in instrument calibration from response law;
tests preserve old default outputs, correct evaluation copies and the public interface.
All cases and the six live pairs were selected before observing frozen outcomes.

| Change | Frozen confirmed/terminal-correct | False confirmations |
|---|---:|---:|
| Stable | 2/2 | 0 |
| Response only | 0/2 | 0 |
| Instrument only | 12/12 | 0 |
| Instrument and response | 12/12 | 0 |

The two response-only failures support the old instrument but repeatedly submit
its refuted parent predictor. Three failed confirmations consume 216/240 rows.
The fixed program ignores this counterevidence. All 24 new-instrument cases pass.
Frozen costs: 3552 observations, 6048 measurement, 535704 computation, zero LLM calls.

Live retention: 3 correct, 1 confirmation-budget failure, 2 capacity interruptions.
Live deletion: 3 correct, 3 capacity interruptions. All six confirmed predictions
pass terminal scoring. Retained agents invoke unchanged method source nine times.
One response-only case recovers by bypassing the method and refitting the response
from paid evidence (fresh/terminal MSE 3.84e-18/6.40e-18). The other buys 96 extra
meter rows and refits both parts; validation MSE reaches 3.13e-18 but only 24 rows
remain for 48-row confirmation. Its deleted control succeeds. No method is rewritten.

## Limitations and next step

Five provider failures are capacity, not timeout; raw 3/6-versus-3/6 is not an
algorithmic tie. Their total token usage is unknown; completed-call reports sum
to at least 1,670,387 tokens (cached 573952/1660332 known input tokens, 34.6%). Failed
work and common learning remain in the ledger. This is parameter transfer within
supplied recurrent families; paired cases are not independent population samples.
Next learn and retain a response-revision branch from the parent counterexample,
then freeze and test it under other response calibrations and stable controls.

[Protocol](protocol.json) · [Frozen results](frozen/summary.json) ·
[Live results](live/summary.json) · [Combined record](summary.json)
