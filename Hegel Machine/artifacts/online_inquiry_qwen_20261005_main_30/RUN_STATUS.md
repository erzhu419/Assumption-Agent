# Same-model Qwen inquiry comparison: interrupted, 2026-10-05

The qwen3.8-flash batch used ruoli native Chat Completions with the same scientific prompts, tasks, research limits and scoring as the previous GPT study. Thirty source-to-target chains were registered per arm, 180 task slots. Nine slots were attempted: one Code source passed fresh confirmation and 256-point independent evaluation, six ended in infrastructure failure and two in malformed tool output. The remaining 171 slots were not attempted. No complete source-to-target chain or paired target comparison exists.

| Target outcome | Code/ReAct | Textual memory | Hegel |
|---|---:|---:|---:|
| Registered | 30 | 30 | 30 |
| Attempted | 1 | 0 | 0 |
| Confirmed and independently correct | 0 | 0 | 0 |
| Service failure | 1 | 0 | 0 |
| Unattempted | 29 | 30 | 30 |

Five service failures returned HTTP 500 model_unavailable; one was a read timeout. The automatic stop prevented new dispatch after three consecutive model_unavailable chain completions and retained in-flight results. The process has exited. No additional live batch was started.

The two format failures are distinct: Hegel returned a write action with flat action/path/code fields instead of arguments_json; textual memory returned two tool calls instead of one. The frozen runtime ended those tasks. After this batch, the common runtime was corrected to expose known-usage format errors as charged feedback, allowing a new decision within the original call/token limits. It executes none of the malformed response's actions. HTTP errors, empty responses and missing usage remain terminal. Twenty-nine focused tests pass; this correction has not been rerun on the external model and does not alter this batch's results.

The sole successful source, Code 79101013, identifies y=c143 from paid evidence and obtains MSE 0 in fresh and terminal evaluation, using 22 observations and eight calls. Its optional least-squares branch incorrectly unpacks four return values into two and catches that exception; the actual correct predictor is its explicit c143 formula. This is not evidence of successful least-squares fitting. Generated code is retained unchanged.

There were 40 API calls and at least 198,600 reported tokens; six failed calls lacked usage. All 34 saved Chat responses report qwen3.8-flash. Compatibility probes are separate: the successful configuration uses native tools without strict, named tool_choice or parallel_tool_calls parameters. The failed combined configuration does not identify which individual parameter caused failure. No GPT subscription, automatic retry or model fallback was used. GPT results are not pooled with this cohort.

Evidence: [summary](summary.json), [frozen protocol](protocol.json), [successful source and interrupted target](79101013/code_react/result.json), [successful native-tool check](../online_inquiry_qwen_connection_20261005/chat_native/connection.json).
