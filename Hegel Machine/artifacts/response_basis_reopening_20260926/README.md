# Response-basis reopening: frozen six-stage study

Twelve new apparatus roots (9984 series), six stages, four arms; all **288 learning stages finished before 288 terminal evaluations**. All processes used `PYTHONHASHSEED=0`. Task, source and budget details are fixed in [the protocol](fresh_replication/protocol.json); the final 0.117 runtime is frozen. No model API was used. Empty is an offline search agent, not a vanilla LLM.

Every cumulative stage receives the entire preceding KnowledgeLibrary through its complete checkpoint. Stage caps alternate 256/384 observations, 180000/240000 computation and 3072/4608 measurement; unused resources do not enlarge later caps. Returning mechanisms buy new observations, without old-task resumption.

| Arm | Pass / 72 | Completed and pass | False complete | Mean MSE | Observations | Measurement | Computation |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 51 | 44 | 0 | 4.96104 | 13283 | 132228 | 9532539 |
| cumulative | 62 | 54 | 0 | 0.00187423 | 13081 | 126552 | 7664855 |
| fixed_source | 38 | 21 | 0 | 0.0206596 | 16854 | 182376 | 14622314 |
| empty | 48 | 31 | 1 | 0.0058224 | 17318 | 199176 | 14095488 |

Table costs cover successor learning. The 12 genuine previously learned source libraries cost 3422 observations / 37896 measurement / 2772605 computation, counted once per seeded arm, not once per stage. Empty has no source cost. Revised cumulative source-plus-learning computation is 10437460, versus old cumulative 12305144, fixed source 17394919 and empty 14095488. CPU in JSON covers research execution only, excluding checkpoint I/O and library construction.

Revised versus old cumulative: **14 lower-loss / 0 higher-loss / 58 tied** stage pairs; learning computation -19.59%, observations -1.52%, measurement -4.29%. Including source costs, computation falls 15.18% versus old cumulative and 25.95% versus empty. Revised versus fixed source is 28/4/40, versus empty 24/7/41. Whole six-stage terminal success improves from 3/12 to 6/12 roots; fixed-source and empty each have 1/12. These are finite-family results, not indefinite lifelong or open-world performance.

Old → revised passes by stage: source entry 6→6; range change 9→9; response change 6→12; instrument change 9→12; source return 10→11; changed-mechanism return 11→12. Revised response/instrument/changed-return successes are all completed. No arm internally completes and passes every stage of an entire chain. [Full paired summary](fresh_replication/summary.json).

## Development records

- [Initial development](development/summary.json): old and revised each 6/12 passes, but revised mean MSE rose from 0.0240 to 122.85. The negative result is preserved.
- [Single-task confirmation fallback](confirmation_fallback_replay/summary.json): same paid observations, 912 extra computation; known root03 source-return MSE 1474.01→0.00280455. State stays unresolved and confirmation_loss is cleared. This is a diagnostic, not independent evidence.
- [Repaired development](development_repaired/summary.json): old/new 6→10 passes, 4→8 completed-and-passed, 4 wins/0 losses/8 ties. Twelve identical fixed-hash baseline checkpoint/evaluation pairs were reused; 36 new stages and terminal evaluations ran. No formal outcome changed the final code.

## Next mainline

[Three remaining range failures](next_gap.json) follow instrument-input revision with 127503–131484 computation spent on refuted interval conditioning and no relation experiment. Fixed-source controls solve those same tasks. Next: choose jointly between response refitting, range reopening and paid discrimination after instrument repair, reserving experiment resources. Remaining failures are six source-entry tasks, three range changes and one source return.
