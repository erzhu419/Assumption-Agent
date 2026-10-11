# Fixed extension — negative result

All 20 continuations and 280 terminal scores completed using frozen 0.179.1 versus 0.178.0. Extension passes 117/140 versus 118/140. Combined with the corrected two-root development: 140/168 versus 141/168, MSE 0.02240106 versus 0.02124524; last seven positions 74/84 versus 75/84. Two tasks failed newly and one recovered.

Candidate adds 264 rows and 189,052 compute overall. Pair batches increase 111→120 and true distinction 19→20, while positive-gain batches remain 18. Shared residuals were selected 34 times; local 86 times, including 82 unfunded validation attempts that retained the local plan.

`combined_summary.json` holds the complete comparison; `compact_results.json` holds the extension alone. Candidate is not adopted. Production remains parent-controlled 0.178.0. The original 0.179 negative result and frozen 0.179.1 are retained. No additional studies or tuning are planned.

CPU was 541.22 versus 545.22 seconds overall, with the development baseline reused; the concurrently run extension was 453.67 versus 452.62 seconds. These are individual observations, not a repeated CPU benchmark.

Whole twelve-root primary comparison:

| Metric | 0.178 baseline | 0.179.1 candidate |
|---|---:|---:|
| Passed | 141/168 | 140/168 |
| MSE | 0.02124524 | 0.02240106 |
| Last-seven passed | 75/84 | 74/84 |
| Observation rows | 92,424 | 92,688 |
| Measurement cost | 90,816 | 91,080 |
| Compute cost | 46,685,963 | 46,875,015 |

The original 0.179 four-continuation negative study is in the parent directory; the 0.179.1 two-continuation correction development is in `../correction/`; this directory contains only the ten-root extension. Frozen sources remain in `../runtime/src`, `../budget_reserve_runtime/src`, and `../baseline_runtime/src`; test sources and XML remain in the parent artifact.

The two new failures missed repairs: r01 p10 never received the baseline second visit/counterexample; r10 p5 was not revisited. Neither trace shows a successful repair being withdrawn (`new_failures_diagnosis.json`).
