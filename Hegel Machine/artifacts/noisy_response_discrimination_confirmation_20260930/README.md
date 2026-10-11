# 0.132.1 confirmation revision: negative, not adopted

Neither the first 0.132 controller nor this 0.132.1 controller is adopted as the default. The chosen default remains 0.131; the experiment helper remains callable. Both complete prospective runs are preserved, with no further experiments or result-driven retuning in this batch.

The first round remains unchanged at `artifacts/noisy_response_discrimination_20260930`: its new cumulative arm has mean-response MSE +3.27% and research CPU +28.43% relative to its same-cohort 0.131 baseline. This second round uses different apparatus; its conclusion is separately paired against 0.131, not a causal comparison between rounds.

Offline `proposer=search`, with no external model calls. The second-round baseline directly equals 0.131 across 451 Python files. Candidate `fresh_replication/runtime/src` preserves the complete frozen 0.132.1 controller (452 Python files), copied exactly from the finalized development runtime and production at freeze. The first frozen controller remains at `../noisy_response_discrimination_20260930/fresh_replication/runtime/src`.

Development uses seen apparatus 9914301/03/08, three arms × 18 tasks. Matching 0.131 checkpoint/visit costs are reused; terminal copies occurred only after all development learning. New cumulative completes 18/18 but mean-response MSE is +24.09%; the previous root03/window7 improvement is lost. Its retained-parent trace is `development/root03_window7_held_parent_trace.json`. The raw one paired win is only a 7.94e−18 floating-point difference.

Formal `fresh_replication/protocol.json` was fixed before learning: new apparatus 9914501–12, three arms × 72 tasks, windows 3/5/7/2/9/return3 and final-response Gaussian noise .01. Per-task limits are 240 observed rows and 240,000 compute; cumulative chain pools are 1,440/1,440,000 with natural revisits. Independent resets the complete library per task. All 36 chains finish before common 4×64 terminal scoring. Both learning dispatchers and finalizer exit naturally with code 0; 216 tasks, 864 trajectories and 55,296 measured terminal rows are complete. Private noiseless targets exist only in scoring.

| Second-round arm | Completed | Mean-response MSE | Research observations | Compute | Research CPU (s) |
|---|---:|---:|---:|---:|---:|
| 0.131 cumulative | 65/72 | 1.24588e−5 | 15,961 | 4,044,907 | 467.75 |
| 0.132.1 cumulative | 66/72 | 1.66372e−5 | 15,565 | 3,810,131 | 381.01 |
| 0.132.1 independent | 70/72 | 1.93782e−5 | 14,816 | 2,750,975 | 137.90 |

New cumulative vs 0.131: mean-response MSE +33.54%, research CPU −18.55%, observations −2.48%, compute −5.80%; apparatus 1 improving/11 worsening. Raw task pairs are 7 wins/30 exact ties/35 losses. Including terminal, old/new cumulative CPU is 472.400/385.853s, observations 34,393/33,997 and compute 4,081,771/3,846,995. The additional completion does not imply better prediction.

Directed trials buy 1,438 rows across 51 trials: 33 supported, 5 refuted, 13 unresolved. Ordinary paired confirmations support only 4/46 and leave 42 unresolved; 42 retain the parent, including 30 previously supported by the directed trial. Of those 42 unresolved ordinary pairs, 37 have positive estimated gain. This conservative allocation blocks most revisions without establishing useful prediction gains. `fresh_replication/research_attribution.json` and `fresh_replication/ordinary_confirmation_uncertainty.json` retain the recorded evidence separately from adoption.

## Limitations

The controlled response family does not establish open-world capability or external LLM/Code superiority. Cumulative vs independent jointly changes knowledge, pooled budget and revisits: MSE improves 14.14%, but research CPU is 2.76×, compute +38.50%, observations +5.06%, and completion is 66 vs 70. Stable net knowledge benefit is unestablished. Research CPU measures `session.run`, excluding checkpoint serialization/reload; every source/research/terminal charge remains reported.

Next core: allocate finite observations between fitting, targeted trials and representative confirmation, and keep uncertain positive research evidence available for continuation. No further confirmation gates or formal-data threshold changes are justified by these two negative rounds.
