# Explicit evidence continuation: completed development retest

One new apparatus (seed 98106007) supplies finite and recurrent continuations.
All three arms use subscription gpt-6-luna and identical optional operators;
model-authored `after_observation` steps connect acquisition with paid-data use.

| Arm | Correct sources | Correct targets | Total observations | Calls | Reported tokens |
|---|---:|---:|---:|---:|---:|
| Code/ReAct | 2/2 | 2/2 | 342 | 27 | 595692 |
| Textual memory | 2/2 | 2/2 | 378 | 19 | 437330 |
| Hegel | 2/2 | 2/2 | 360 | 33 | 872403 |

Every accepted model passes 48 fresh rows and independent 256-row evaluation;
there are no false confirmations or provider failures. The 79 calls report
1905425 tokens; 50.34% of input tokens are cache hits. Focused checks passed:
12 operator tests and 19 inquiry/context/runner tests.

The [actual recurrent Hegel trace](mechanism_trace.json) reuses the source plan,
buys two six-step meter experiments and assesses their evidence immediately.
The old finite candidate is refuted (MSE .008481), while the retained recurrent
candidate is supported (7.12e-19). The old response errs at .001031. The LLM
then refits the parent on the supported coordinate, reaching validation MSE
2.88e-19, fresh confirmation 1.18e-17 and terminal MSE 9.26e-18. The finite
target instead successfully reuses the frozen source predictor unchanged.

The recurrent target uses 72 observations against 96 for either control, but
including its source Hegel uses 192 against 174. Across both sequences Hegel
uses 324 measurement units against Code's 450, with more observations and calls.

Limitations: one paired apparatus; new seed, prompt compaction, trajectory split
and continuation interface change together. There is no isolated effect estimate
or overall superiority. Continuations are authored in query arguments, not saved
in the inherited plans. All online questions are roots; the later manifest is
not an executed recursive dependency graph.

Next: learn and retain the complete acquisition-to-revision procedure, transfer
it to another changed scope, and compare with removal of that procedure while
keeping the same predictive knowledge. Do not expand this tied accuracy pilot.
