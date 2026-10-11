# Temporal research with a matched flat control

Both policies confirm and independently solve 60/60 targets: 30 finite-memory
and 30 recurrent cases, with zero false confirmations. All training stops before
terminal evaluation. Each predictor receives 48 fresh confirmation rows and a
separate 256-row terminal sample; the MSE threshold is 1e-10.

| Recurrent task, per case | Recursive | Flat adaptive |
|---|---:|---:|
| Correct | 30/30 | 30/30 |
| Target observation rows | 163 | 159 |
| Measurement units | 57 | 45 |
| Instrument search row-passes | 498 | 410 |
| Response fits | 1 | 3 |

Finite cases use 144 target rows and no meter readings in both arms. Source cost
is 48 rows per case. The arms share apparatus draws, source and initial target
actions, candidate families, fitting functions, probe planner and resource caps.
Flat fits both instrument-conditioned response candidates on 63 paid rows and
selects on the other 48 whole-trajectory rows before refitting all paid data.
The response evidence selects the recurrent coordinate even though the short
meter records alone leave both instrument explanations possible. Flat therefore
does not require the recursive policy's additional four-step meter experiment.

The shared representation repertoire achieves the same accuracy without
separately confirming the instrument child. The earlier causal child-repair
result remains valid; it does not establish an advantage for this organization.
Fifteen focused tests pass. No LLM calls were made. See `plan.json`,
`summary.json`, and per-case programs, acquisitions and decision traces.

Limitations: these are bounded offline policies. The common computation cap
counts instrument search only; response fitting is reported separately and
no total-compute advantage is claimed. Current-task LLM performance and method
invention remain open.

Next: expose the same individual research operators to the common LLM arms,
testing their selection and reuse. A successful parent prediction may be
accepted while an instrument interpretation remains unresolved.
