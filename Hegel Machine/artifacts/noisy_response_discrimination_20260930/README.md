# 0.132 noisy response discrimination

Offline `proposer=search`; no external model calls. The 0.131 baseline directly equals the prior formal runtime (451 Python files); the formal 0.132 runtime directly equals the finalized development snapshot and production (452 Python files).

Development uses seen apparatus 9914301/03/08, three arms × 18 tasks. Matching old cumulative checkpoints and visit files are reused from 0.131; old terminal outputs were copied after all development learning. New cumulative retains 18/18 completion, mean-response MSE −20.55%, research CPU −55.39%, observations −4.93%, compute +10.93%. Root03/window7 falls from 1.19542e−4 to 9.12856e−6; root01/window2 worsens from 5.77313e−6 to 4.49062e−5. `development/root01_window2_negative_trace.json` records the inconclusive directed trial and stopping with 55 unspent rows.

Formal `fresh_replication/protocol.json` was fixed before learning: new apparatus 9914401–12, three arms × 72 tasks, windows 3/5/7/2/9/return3, final-response Gaussian noise .01. Per-task budgets are 240 observations and 240,000 compute; cumulative chain pools are 1,440/1,440,000, with natural revisits. Independent resets the complete library per task. All 36 learning chains finished naturally before common 4×64 terminal scoring; 216 tasks, 864 trajectories and 55,296 measured terminal rows are complete. Private mean targets exist only in terminal scoring.

| Formal arm | Completed | Mean-response MSE | Research observations | Compute | Research CPU (s) |
|---|---:|---:|---:|---:|---:|
| 0.131 cumulative | 59/72 | 1.46084e−5 | 15,639 | 4,598,996 | 509.53 |
| 0.132 cumulative | 59/72 | 1.50855e−5 | 14,778 | 4,988,900 | 654.39 |
| 0.132 independent | 68/72 | 1.63611e−5 | 14,651 | 3,079,756 | 169.04 |

New cumulative vs old: mean-response MSE +3.27%, research CPU +28.43%, observations −5.51%, compute +8.48%; 4 task wins/59 exact ties/9 losses, and 3 apparatus wins/5 ties/4 losses. Predictive noisy MSE is 1.13456e−4→1.13911e−4. Including terminal, old/new cumulative observations are 34,071/33,210, compute 4,635,860/5,025,764 and CPU 518.75/659.25s. Formal evidence does not establish an overall improvement.

New cumulative buys 46 directed trials (1,365 rows): 30 supported, 7 refuted, 9 unresolved; 30 supported refits are retained on a separate ordinary confirmation. Successor offers increase 107→152, screening compute 795,281→952,294. Cumulative vs new independent improves mean-response MSE 7.80% (18 wins/42 ties/12 losses; apparatus 7/1/4), but costs 3.87× research CPU, 61.99% more compute and 0.87% more observations, while completing fewer tasks. Stable net knowledge utility is not established.

## Limitations

This controlled response family does not establish open-world capability or external LLM/Code superiority. Cumulative vs independent jointly changes retained knowledge, budget pooling and revisits. Research CPU measures `session.run`, excluding checkpoint serialization/reload; all source/research/terminal charges remain reported. The worst formal pair, root01/window9, worsens from 1.56737e−6 (unresolved) to 5.29121e−5 (completed): directed support is retained despite inconclusive ordinary confirmation, and terminal accuracy worsens. Its compact trace is `fresh_replication/root01_window9_negative_trace.json`.

Next core: additional paid evidence when directed gain remains uncertain, and prediction coverage beyond the single selected separating trajectory. No formal outcome changes production, thresholds, seeds or budgets in this batch.

Reproduce phases with `python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/noisy_response_discrimination_20260930/<group> --phase <learn|evaluate|summarize> --workers <n>`. `compare_results.py`, `attribute_cpu.py`, `attribute_research.py` summarize finished artifacts only.
