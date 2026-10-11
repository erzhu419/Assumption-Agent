# Historical methods in a live LLM researcher

**One explicit cross-task method reuse succeeds; aggregate efficiency is mixed.**
The same subscription `gpt-6-luna` runs two matched window/event pairs. Only the
method book differs, taken unchanged from the measured v0.76
`temporal_method_reuse_20260913/development_v1/runs/151_temporal_methods.json`.
Current calibration, supplied candidates, tools and limits are identical.

| Target totals | Retained book | Removed book |
|---|---:|---:|
| Confirmed and terminal-correct | 2/2 | 2/2 |
| Observations | 252 | 336 |
| Measurement units | 468 | 432 |
| Charged computation | 25,946 | 23,540 |
| LLM calls | 9 | 4 |
| Reported tokens | 283,230 | 173,825 |

The retained event agent calls the old method from `100154:event:0`. The planner
rebinds its control from 0.035735 to 0.232564 under the new calibration and threshold,
skips fresh synthesis, and proposes four observations. Actual responses
`[0,0,1,-1]` uniquely select the event candidate; measured feedback records
`use_program_experiment_method`, `cross_task=true`. Its frozen predictor also
passes terminal evaluation. Other targets design their own informative queries.
The retained event agent first spends work fitting all-zero observations.

The four targets use 457,055 reported tokens, with 180,736/451,209 cached input
tokens (40.1%). Historical training cost remains separate: 648 observations,
1,944 measurement units and 36,607 computation. Results are in
[run1/summary.json](run1/summary.json); all research ends before terminal scoring.

## Limitations

Two development pairs with supplied candidates establish actual use, not an
average advantage or new hypothesis invention. Book contents are also visible
in the prompt, so language-level influence is not isolated. The initial launch
stopped before any model call because its case directory was absent; `run1` is
the completed run. The earlier source-learning pilot remains separately recorded.

The v0.93 research-action connection and live follow-up are now completed:
[research-choice results](../temporal_historical_actions_luna_20261006/RUN_STATUS.md).
