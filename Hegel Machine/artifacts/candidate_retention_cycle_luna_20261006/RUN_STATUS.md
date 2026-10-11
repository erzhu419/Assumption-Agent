# Live hypothesis retention through change and return

Two new apparatus instances; one live Luna source per instance, then retained-history and latest-only branches, each with a changed response and the original response's return. All 10 research episodes ended before terminal evaluation. Every episode passed fresh confirmation and its separate 128-row terminal test. No service failures, episode replacements or episode reruns occurred.

| Outcome / whole-sequence cost | Retained history | Latest only |
|---|---:|---:|
| Target episodes confirmed and correct | 4/4 | 4/4 |
| Returns submitting the unchanged original predictor | 2/2 | 0/2 |
| Calls on the two returns | 2 + 2 | 11 + 8 |
| Calls including common source acquisition once | 30 | 42 |
| Reported tokens including source acquisition | 697,061 | 1,188,554 |
| Paid observations including source acquisition | 576 | 600 |
| Host candidate-row scoring units including source | 3,432 | 5,568 |

Each condition includes the same 11-call, 192-observation source acquisition. Physical usage was 61 calls and 1,603,749 reported tokens; all calls reported usage. Cached input was 862,208 / 1,588,312 tokens (54.28%). Python execution totals including sources were 6.43 / 9.33 seconds; summed research wall times were 391.02 / 568.59 seconds. These are episode sums with shared sources counted once per condition, not elapsed campaign times.

In apparatus `791006001`, retained return call 1 compared original and changed predictors: paid MSE `2.46e-29` / `0.38385`. Call 2 submitted and freshly confirmed the original. In `791007010`, the original predictor had failed on changed evidence (MSE `0.42888`); return call 1 evaluated it successfully, and call 2 confirmed it. Both used unchanged archived code, dependencies and original fitting data. The strict `recommendations_followed` counter is zero because the first Agent submitted the original archive path instead of the suggested equivalent snapshot path; the recommended candidate was actually adopted. The second return used direct evaluation, without `research_choice`.

Latest-only's first return wrongly submitted the changed predictor; fresh MSE `0.27969` exposed counterexamples, after which it learned and confirmed a new predictor. Its Python syntax error and the other retained branch's nonexistent-file submission remain in the traces. Neither became a false confirmation. Eight new tests cover the live sequence, fitting context, real branch continuation, declared dependencies and changed-file attribution; 19 related existing tests also passed during implementation.

## Limitations / next step

This two-instance, same-family pilot demonstrates autonomous hypothesis reuse with lower observed cost; it does not estimate population accuracy or stable long-horizon benefit. Latest-only tests historical retention's incremental value. Host scoring units omit arbitrary Python fitting arithmetic. Archived `fit()` callbacks may rebuild parameters on their original rows; neither retained return fits on the new task's labels. Successful prediction does not confirm the Agent's structural explanation: latest-only attributed an old-coefficient failure to insufficient history and later used nine lags, although a three-lag model also solved that task. The next cognitive target is distinguishing parameter failure from structure or research-method failure, then testing evidence-driven method revision.

[Frozen protocol](protocol.json) · [Complete summary](summary.json) · [Runner](../../scripts/run_candidate_retention_cycle_llm.py). Per-episode `training.json`, `published_candidate.json` and `evaluation.json` retain actions, provenance and terminal results; `source_at_run/` preserves the executed implementation.
