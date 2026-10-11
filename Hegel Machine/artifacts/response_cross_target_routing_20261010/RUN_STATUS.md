# Cross-target response routing: frozen paired execution

All 12 executions finished. Task completion improved from **2/6 to 3/6**, with fewer world actions (1,022 → 976) and more research compute (7,180,504 → 8,678,260). This is a development-panel result for one host-controller predicate change.

Both arms used the same 16-declaration program, immediate-completion fixture and original limits: 512 actions, 256 procedure steps, 240 Python seconds per task, 5 seconds per call, 3,600 task seconds and zero LLM calls. The treatment replaced only `_point_needs_response`; the full old/new bundles and protocol were saved before execution. No assigned program changed during execution. Costs below are old routing → cross-target routing.

| Condition | Completion | Actions | Research compute | Procedure steps / termination |
|---|---|---:|---:|---|
| cubic original low | fail → fail | 213 → 203 | 944,263 → 2,360,928 | 256 limit → 256 limit |
| cubic shifted low | fail → success | 211 → 175 | 1,615,721 → 1,696,812 | 256 limit → 221 complete |
| cubic original high | fail → fail | 217 → 217 | 504,855 → 504,855 | 256 limit → 256 limit |
| cubic shifted high | success → success | 185 → 185 | 2,311,615 → 2,311,615 | 228 complete → 228 complete |
| square shifted | success → success | 101 → 101 | 555,477 → 555,477 | 141 complete → 141 complete |
| aliased shifted | fail → fail | 95 → 95 | 1,248,573 → 1,248,573 | 119 stopped → 119 stopped |

For both low conditions, target 5 now enters response research on the **next procedure step after its first point refutation**: original at step 162 / observation count 119; shifted at step 215 / count 170 (step indices are zero-based). Their old arms made no target-5 response call. The shifted condition selects 4450 and achieves actual activation at 4452 during tuning, completing the task; the requested 4450 response test remains pending, so this is operational completion rather than confirmation of that exact point prediction. The source target's earlier requested point remains refuted despite its successful operational recovery.

The original-low arm reaches one new joint revision, tests 5901 once, and genuinely refutes it at step 192. Seven subsequent `family_research_needed` returns are followed by broad-pool member testing, consuming another 56 world actions and 1,086,654 compute before the step limit. The remaining issue is **research selection after a joint prediction fails**, rather than a lost routing connection or duplicate joint test. Historical conditional support-domain and branch-value mechanisms are candidates for the next repair; these compact traces do not establish the complete remaining feasible domain or that further narrow enumeration would help.

The other four pairs have identical actual action sequences. Original-high has no prior target response research to transfer and retains its old late response route (step 240, after 12 point refutations). Square already routes after one refutation through the old same-family condition. New joint-revision counts are old 0 / treatment 1. All executions retain 16 declarations and unchanged assigned source; completed and pending trials contain **zero `authored_experiment`** records. The later authored-setting routing correction was made after these executions and is covered by its focused regressions, not this frozen panel.

## Limitations

These six conditions reuse a development panel of parameter interpolation and existing controls; they do not establish unseen-structure discovery or an LLM-learned strategy. Both arms use the corrected immediate-completion fixture, so the earlier Exit-only panel is not a matched baseline. All failures are retained; no budget extension or outcome-driven rerun was made.

Evidence: [protocol](protocol.json), [paired results](paired_summary.json), [routing events](routing_summary.json), [full compact summary](summary.json). Each case/arm folder retains its research trace, actual action sequence and final public observation.
