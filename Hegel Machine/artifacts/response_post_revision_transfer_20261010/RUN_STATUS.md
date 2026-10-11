# Post-revision research selection: six frozen executions

All six executions finished. Against the saved `cross_target_routing` results, completion remains **3/6**, while world actions fall **976 → 921** and research compute **8,678,260 → 7,765,634**. The repair removes costly unsuccessful continuation; it does not solve another condition.

| Condition | Completion, before → after | Actions | Research compute | Procedure steps / termination |
|---|---|---:|---:|---|
| cubic original low | fail → fail | 203 → 148 | 2,360,928 → 1,423,963 | 256 limit → 196 stopped |
| cubic shifted low | success → success | 175 → 175 | 1,696,812 → 1,696,812 | 221 complete → 221 complete |
| cubic original high | fail → fail | 217 → 217 | 504,855 → 505,839 | 256 limit → 256 limit |
| cubic shifted high | success → success | 185 → 185 | 2,311,615 → 2,311,615 | 228 complete → 228 complete |
| square shifted | success → success | 101 → 101 | 555,477 → 578,832 | 141 complete → 141 complete |
| aliased shifted | fail → fail | 95 → 95 | 1,248,573 → 1,248,573 | 119 stopped → 119 stopped |

Original-low still refutes the joint prediction 5901. The repaired controller then compares the existing reference pool over the full conditional target domain and stops without executing the previous broad-pool fallback: **55 fewer actions and 936,965 less compute**. Its final assumed support still contains 225 grid points. Four reference candidates each have zero guaranteed domain reduction across both outcomes; the target candidate guarantees only eliminating its own point. This is a request for different research, not evidence that every hypothesis is false. The next substantive step is to construct a more informative measurement or revise the research assumptions. [The separate fixed-state diagnosis](../response_joint_refutation_diagnostic_20261010) supports that distinction.

The other five actual action sequences are identical to baseline. Extra comparison costs are retained, including 23,355 on square and 984 on original-high. All six assigned programs stayed unchanged with 16 declarations, zero LLM calls and zero authored experiments. One joint-revision scope was exercised, on original-low.

## Limits and artifacts

This reuses six development conditions and a historical matched baseline with the same immediate-completion fixture and original limits. Only `_extend_hypotheses` was replaced in the source bundle; the two revised runtime modules were archived before execution. No budget increase, rerun or outcome-driven tuning occurred. The frozen stop message still says hypotheses are exhausted; the saved comparison explicitly remains `family_research_needed` with nonempty conditional support. A later wording correction does not change these results.

[Protocol](protocol.json) · [Paired results](paired_summary.json) · [Diagnostic summary](diagnostic_summary.json) · [Costs](summary.json). Each case retains its full current comparison, research table and tests in `current_state.json`, with historical comparisons compacted.
