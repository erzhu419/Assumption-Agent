# Paid observed-condition decoding

The completed 0.137.0 baseline is frozen as 459 Python source files. Initial 0.138.0 development is 461 files, corrected development is 463, and [formal runtime](fresh_replication/runtime/candidate_freeze.json) was copied unchanged from corrected frozen source. All research is offline; external model calls are zero.

[Runner](../../scripts/run_observed_condition_decoding_study.py) uses three raw controls and one purchased random process reading. The public instrument description states its actual calibration. The reading never appears among command bounds. Re-reading the same experiment preserves its physical realization but charges observation and measurement again; new experiments sample fresh perturbations. Saved purchases restore that exact sequence.

[Known meter development](development_known_meter/protocol.json) reproduces the existing example. [New development](development/protocol.json) fixes apparatus 9915501–9915502. [Formal protocol](fresh_replication/protocol.json) fixes twelve new apparatus 9915601–9915612: seven stages and 252 arm-tasks. Stages cover plain W3, historical conditional W3, current conditional W5, changed historical W7, renamed/scaled current W7, return to historical W3, and plain W5. Both branches have a continuous instrument contribution; TaskSpec exposes no true threshold, window or response coefficients.

Each task receives 320 observations, 320 measurements and 320000 computation. Cumulative chains may spend only unused visited-task funding on natural revisits, within 2240/2240/2240000 chain totals. Independent resets knowledge and unused resources each task. All learning and revisits ended before common four 64-row terminal trajectories; the MSE <=1e-6 criterion is unchanged.

[Known meter summary](development_known_meter/summary.json): old falsely completed at MSE 0.113014; both new arms passed at 1.48e-19. [First development](development/summary.json) retains cumulative 7/14 and independent 8/14, including lost W7 structures and a wrong W10. [Corrected development](development_corrected/summary.json) solved 7/14 cumulative and 9/14 independent after replacing single-row reset refinement with a paid fixed-background trajectory followed by varied-control response fitting and explanation competition. All 28 corrected new structures matched the protocol; false completions remained 6 and 4. All initial source and negative results are preserved.

[Development negative classification](development_corrected/negative_classification.json) reads existing reports only: ten models had nearly zero paid validation/confirmation loss but failed independent scoring with finite threshold intervals; two were already refuted on paid evidence and remained unresolved. No terminal rows were reconstructed or rescored for this classification.

[Formal summary](fresh_replication/summary.json): all 36 apparatus-arm sequences and natural revisits finished, followed by one common terminal evaluation. Costs below cover complete learning and revisits, excluding terminal scoring. Correct completions equal solved tasks; claimed completions are 59, 82 and 77, respectively.

| Arm | Solved | False completed | Mean terminal MSE | Observations | Measurements | Computation | Research CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old cumulative 0.137 | 30/84 | 29 | 3.32071 | 22713 | 13273 | 26684193 | 1413.43 |
| New cumulative 0.138 | 44/84 | 38 | 0.160402 | 14463 | 12663 | 13256796 | 510.32 |
| New independent 0.138 | 44/84 | 33 | 0.738233 | 18903 | 10839 | 13687526 | 387.75 |

Conditional tasks including return solved 6/60, 20/60 and 20/60; return alone solved 4/12, 6/12 and 4/12. All 24 plain controls passed in every arm. Natural revisits were 28, 5 and 0. New cumulative versus old had 44 paired improvements, 29 ties and 11 deteriorations; versus independent it had 26 improvements, 31 ties and 27 deteriorations.

Cumulative versus independent learning uses 23.49% fewer observations and 3.15% less computation, but 16.83% more instrument measurements and 31.61% more CPU. Mean error is lower, while solved counts are identical and false completions are more frequent. All formal failures are retained without source, seed, budget or criterion adjustment.

## Limitations

Correct window and temporal condition placement do not guarantee threshold accuracy or reliable completion. All 84 cumulative programs retain the specified window/condition placement, yet 38 claimed completions failed independent scoring. The development evidence supports finite threshold uncertainty; no claim of a row-by-row terminal attribution is made. Four independent unresolved cases differ in their saved window form: root01 and root12 scaled W7, root03 historical W7 (selected W8), and root04 return W3. The largest is root01 scaled W7, reduced to a sparse affine explanation, at MSE 17.954. The finite-history alternatives were not separately reconstructed to classify every history dependence.

These are finite quadratic-response laboratories with one calibrated uncontrollable instrument. Independent is a knowledge-reset control, not vanilla Code or LLM. Reliable long-term knowledge benefit and open-world capability remain unestablished.
