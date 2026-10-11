# Selective preservation: completed

Both the strong refit-both baseline and the learned program pass56/56 with zero
false confirmations:44regression controls and12new calibration cases. All8earlier
regressions are repaired. The program tests the old prediction first; failure
leads to new meter evidence, then instrument retention or reopening. Actual
choices are20retentions,32reopenings and4unchanged-predictor reuses. No learned
source was hand-edited, and transfer uses no LLM calls.

| Frozen cost | Baseline | Learned |
|---|---:|---:|
| Observations | 12576 | 12960 |
| Measurement | 23328 | 23328 |
| Computation | 1169268 | 754464 |

Computation falls35.48%;384extra observations come from testing the old predictor
on8initially changed-instrument tasks. On the12new cases, observation/measurement
are equal and computation falls30.81%. Authoring uses11calls/732491tokens
(32.54%cached input),1656observations/3240measurement/71427compute. Including the
preceding unsuccessful selective phase:19calls/1182549tokens. Its36/44result is
preserved. The new failed replay additionally costs216/648/72respectively.

See `summary.json`, `study/learning/result.json`, and the saved protocol. Seventeen
related tests passed. Remaining scope: fixed experiments, supplied model families
and a disclosed checkpoint; ambiguous or noisy evidence still needs adaptive
experiment selection. Instrument refutation does not identify response-law change.

Metadata note: saved protocols inherited a stale `supplied_trace_origin` from the
preceding phase. Actual author feedback uses their correct `counterexample` path
(`counterexample/episode_00/training.json`); the runner now drops that stale field.
