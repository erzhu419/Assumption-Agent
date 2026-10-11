# Complete knowledge handoff through six task generations

The comparison carries entire serialized `KnowledgeLibrary` objects through six-stage chains. Each stage is a real `UnifiedResearchSession` with one new task and writes a compressed complete session checkpoint; the next cumulative stage restores its knowledge from that file. This isolates sequential knowledge handoff from old-task revisit scheduling. Returning a previous mechanism buys new observations and does not restore its old evidence bank.

Stages: new apparatus under the source response/range; changed relation range; changed response; changed instrument names/wiring; return to the first mechanism/interface; return to the fourth mechanism/interface. Observation caps alternate 256 and 384 (computation 180000 and 240000; measurement 3072 and 4608). Unused resources do not enlarge later stage caps. All arms receive the same successor-task caps and physical seeds. Prior source cost counts once for each seeded arm and is reported separately; the empty arm has no source cost, so the empty comparison is not equal total lifetime investment.

The four formal arms are old 0.115 cumulative, revised cumulative, revised fixed source, and revised empty library. The last is an offline search agent, not an external vanilla LLM. Twelve existing real learned source libraries are reused. Formal successors use 12 newly declared apparatus roots (9983 series), with six stages each: 288 learning stages, then 288 independent terminal evaluations. Terminal controls are fixed at four 64-step trajectories. Unresolved programs are also scored. Terminal information never enters learning or library updates.

`development/` retains the first two-root attempt. Its old cumulative, fixed-source and empty controls use 0.115. Revised cumulative stage 0 succeeded for both roots; stage 1 then failed because a new draft claim lacked `method_source_ids`. The traceback and successful checkpoints remain. Those failed stage-1 attempts have no complete research accounting and are excluded from paired scientific outcome/cost claims; this missing failed-attempt cost is retained as a development limitation. Repaired development uses its own frozen runtime and preserves the controls already purchased under identical tasks.

`fresh_replication/protocol.json` fixes all 12 new apparatus chains before formal learning. `summary.json` reports every returned program, stage and stream outcomes, completed-versus-terminal-success separation, all three resource costs, CPU, and reused source costs. Formal effects are interpreted only for this finite six-stage family, not as indefinite lifelong or open-world learning.

Repaired development completed 48 learning stages and 48 terminal evaluations. Old and revised cumulative arms each passed 11/12 programs (9 completed-and-passed, no false completions); fixed-source and empty each passed 10/12. Revised versus old cumulative was 0 wins / 1 loss / 11 ties at absolute MSE tolerance 1e-7, with 1675 fewer computation units but 21 additional observations. This development result does not establish independent performance benefit from draft transfer. All six-stage results, including the failed response-change task, remain in `development_repaired/summary.json`.

After the formal runtime was frozen, released code changed draft ID allocation to avoid reusing an existing ID after explicit library deletion. This protocol never deletes drafts, so it leaves its declared decisions unchanged; the formal runtime stays frozen.

CPU seconds cover `session.run` research execution and exclude checkpoint I/O and library construction. Observation/measurement and metered research computation are reported separately. Same-input 0.115 development controls showed small candidate-order differences between processes; the protocol fixes physical seeds but does not fix Python process hash ordering. Small single-run cost differences alone are not stable evidence of a draft-transfer advantage.

## Formal outcome

All 288 learning stages and then all 288 terminal evaluations completed. Costs below cover successor learning only; reused source cost is 3422 observations, 37896 measurement, and 2772605 computation per seeded arm (12 sources together), charged once.

| Arm | Terminal pass / 72 | Completed and pass | False complete | Observations | Measurement | Computation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 45 | 34 | 0 | 13318 | 135912 | 11144796 |
| cumulative | 43 | 32 | 0 | 13667 | 140676 | 11426135 |
| fixed_source | 38 | 17 | 0 | 16748 | 182544 | 14729908 |
| empty | 51 | 28 | 1 | 17407 | 200820 | 14254161 |

Draft transfer was exercised: 11 drafts preserved, 88 replay/assessment events, and 10 promoted to concepts after confirmation. It did not improve net performance: revised cumulative versus old cumulative had 12 lower-loss, 16 higher-loss and 44 tied stage pairs, with 2 fewer passed tasks and 2.52% more metered computation. Revised cumulative saved 22.43% computation versus repeated source restart, but passed 8 fewer tasks than empty restart. With source costs included, its computation saving versus empty was only 0.39%. No seeded arm passed all six stages on any root; empty passed all six on 2/12 roots.

The main failure is response change: both cumulative arms passed 0/12, fixed source 3/12, empty 9/12; both cumulative arms still passed 12/12 when the original mechanism returned. [next_gap.json](next_gap.json) records two paid research traces showing inherited temporal response terms block discovery of the required current-input response term, even when object roles remain supported. The next mainline is selective response-basis revision under current counterevidence, preserving reusable roles and instrument knowledge.

Full paired outcomes and costs: [formal summary](fresh_replication/summary.json).
