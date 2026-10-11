# Same-model inquiry comparison: partial results, 2026-10-05

The user-selected ruoli / gpt-6.1-sol run stopped after three consecutive service-failed chains; in-flight chains finished and all available terminal evaluations ran. Thirty source-to-target chains were registered per arm, 180 task slots. Sixteen tasks were attempted: nine completed and independently passed, six had service failures and one exhausted its research allowance. The remaining 164 tasks were not started. Previous interrupted batches and connection tests remain separate.

| Target outcome | Code/ReAct | Textual memory | Hegel |
|---|---:|---:|---:|
| Registered | 30 | 30 | 30 |
| Attempted | 3 | 2 | 1 |
| Completed, confirmed and independently correct | 2 | 1 | 1 |
| Service failure | 1 | 1 | 0 |
| Unattempted | 27 | 28 | 29 |
| Successful source–target chains | 2 | 1 | 1 |

All nine completed source/target tasks have fresh-confirmation and 256-point terminal MSE 0, with no false confirmation. There is only one matched Code–Hegel target pair, chain 79101013, and no complete three-arm pair. On this unchanged-structure/unchanged-surface target, Hegel executes unchanged inherited shared/plan.py (nine observations) and shared/model.py (three fresh observations), updating only the evidence and applicability record in shared/knowledge.json. Code also succeeds, with 24 total target observations versus Hegel's 12. This is an actual reuse example, not comparative superiority, structural transfer, method-code rewriting or child recursion.

Code source 79102022 buys all 24 observation slots immediately, leaving none for required fresh confirmation. Nine subsequent finish attempts fail for lack of the three confirmation observations. Research stops at 60,689 reported tokens after 11 calls. This is a research-allocation failure; the later target's empty response is a separate service failure.

The run made 77 API calls with at least 305,535 reported tokens; six calls have no usage. Provider failures comprise five empty completed responses and one stream ending before completion. Of 76 saved Responses payloads, 70 echo the requested instructions exactly, six omit that field, and none differ; all report the model name gpt-6.1-sol. These are returned metadata, not independent verification of upstream routing.

The only pre-run implementation change recognizes empty responses and stops new dispatch after three consecutive service-failed chains, in serial or parallel execution. Tasks, prompts, candidate language, research limits and evaluation are unchanged. Twenty focused tests passed. No automatic retry, model fallback or subscription call was used. The process has exited; no further calls are running.

Evidence: [summary](summary.json), [protocol](protocol.json), [reuse chain](79101013/hegel/result.json), [budget-failure chain](79102022/code_react/result.json). `summary.correct` includes sources and targets and is not a target-only success rate. Missing tasks are not scored as algorithmic failures.
