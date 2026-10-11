# GPT-6-Luna subscription comparison: completed, 2026-10-05

The batch finished at 20:59:27 Asia/Shanghai (exit 0), after approximately 85 minutes. All 90 registered chains have training and independent terminal-evaluation records. Of 180 task slots, 179 were attempted and 169 passed fresh confirmation and the independent 256-point evaluation. No false confirmation was observed. The backend was the existing ChatGPT subscription, `gpt-6-luna / medium`.

| Arm | Correct sources / 30 | Correct targets / 30 | Target failures or missing slots |
|---|---:|---:|---|
| Code/ReAct | 29 | 29 | One target unattempted after source timeout |
| Textual memory | 30 | 29 | One token-budget failure |
| Hegel | 26 | 26 | Three token-budget failures, one model-capacity error |

On the 29 targets attempted by every arm, Code succeeds in 29, textual memory in 28, and Hegel in 25. Hegel therefore shows no overall advantage in this cohort. All submitted models pass terminal evaluation; the deficit is failure to complete research and confirmation.

Three Hegel target failures expose a concrete allocation problem. In `79103031`, an unchanged inherited predictor fails fresh confirmation; after correcting it, the agent has no observations left for another confirmation. In `79127247`, an unchanged inherited plan consumes 22 of 24 observations, leaving fewer than the required three. In `79116148`, a revised 12-row plan runs twice and consumes all 24 observations. Repeated finish attempts cannot recover that observation budget. The fourth target failure is model capacity. Source failures additionally include unproductive model/manifest revisions. Reuse participates in some failures, but does not explain all of them.

The run used 894 model calls: 208 Code, 238 textual memory, 448 Hegel. Reported usage is at least 20,395,181 tokens; two calls lack usage. Cached input totals 14,134,016 tokens, 70.3% of known input tokens; 878 calls report cache hits. All arms shared 300,000 reported tokens, 16 model calls and 24 paid observations per task. Cached tokens count toward the token ceiling. The smaller development pilots and earlier provider/model cohorts remain separate.

The next justified change is research allocation: make experimental plans account for the remaining confirmation budget and stop retrying finish when no confirmation observations remain. This report introduces no new run or modification to the frozen experiment.

Evidence: [summary](summary.json), [target and matched comparison](comparison.json), [protocol](protocol.json), [run metadata](run.json).
