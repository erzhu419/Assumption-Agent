# Anonymous authored revision: completed development pilot

Runtime and task design follow M0/M1 of the 2026-10-11 plan. All six runs used the existing subscription-authenticated `gpt-6-luna` provider; no human edited predictors, research actions or returned measurements during the cohort. The public interface supplied neither mechanism names nor fitted model families. Both arms started with the same 72 paid rows and an empty executable archive.

| Apparatus | Code: terminal / calls / paid rows | Hegel: terminal / calls / paid rows |
|---|---:|---:|
| memoryless | pass / 4 / 189 | pass / 10 / 171 |
| delay | pass / 10 / 576 | pass / 13 / 192 |
| hysteresis | pass / 18 / 546 | pass / 19 / 312 |

All submitted programs passed one 96-row fresh confirmation and independent 512-row terminal prediction on trajectories up to 48 steps. Hegel totals: 42 calls, 675 paid rows, 983,587 reported tokens. Code totals: 32 calls, 1,311 rows, 1,075,278 tokens. Across 74 calls, 59 reported cache hits, totaling 1,029,120 cached input tokens. These three matched development apparatuses do not establish a stable cost or accuracy advantage.

## What actually happened

- Both arms authored a second-lag predictor on case_001 without calling the historical fitting operators. Hegel compared its own lag representations and found the static model insufficient.
- On case_002, Code wrote an internal binary-state predictor at step13, then selected preparation-path experiments at step14. Paid rows258–281 use `[.7,.6×23]` and remain1; rows282–305 use `[-.6,.6×23]` and remain0. Its source stayed unchanged through confirmation and terminal prediction.
- Hegel proposed a history-dependent state question at step7, sampled contrasting paths, wrote its internal-state predictor at step12, and confirmed it at step18. Its explicit question and generated code are in `case_002/hegel/training.json`.
- There were **zero revision-interface calls, zero child questions and zero authored readout returns**. Every actual fresh confirmation passed on its first attempt. Diagnostic-code, JSON and manifest errors were repaired by the model; these are not refutations of scientific hypotheses.
- All six submitted predictors define `predict` but no `fit`. A new-apparatus follow-up separately tests what those saved programs retain.

## Scope and next step

This supports autonomous construction of historical representations and new-path prediction in a finite deterministic setting. Code also generated a useful experiment after writing its model. It does not show autonomous adoption of the newly connected revision/child-return path, unique recursive benefit, or reusable parameter-learning procedures. The next connection should reuse the historical method-edit/practice/frozen-transfer loop to turn the complete source research into an executable successor procedure; declaring the present predictors to be such procedures would overstate the evidence.

The initial `../anonymous_revision_luna_20261011/` startup was stopped for recursive checkpoint collection and remains separate. It consumed four completed Code calls (74,075 tokens) and one interrupted Hegel call with unavailable usage. Those costs are retained and are not part of the fixed six-run cohort above.
