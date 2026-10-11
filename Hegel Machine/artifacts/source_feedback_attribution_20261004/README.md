# Source feedback attribution receipt

Run `PYTHONPATH=src:tests python3 scripts/run_source_feedback_attribution.py`.

The script first earns a real confirmed positive return for `concept_0002`, then freezes eight source offers and four per-task execution caps. The natural four arms retain all returns, delete the bad source, delete the good source, or delete both. All arms use 132 observation rows; execution caps are 24 rows and 120 compute per visit. Screening compute is recorded separately.

The natural masks make the same task choices and held-out losses. The two candidate immediate gains differ slightly, so the scheduler does not need source feedback to choose the better candidate. This is a non-identifying natural attribution result, not evidence that the good source has no contribution.

The `tie_controlled_good_return` section is an explicit controlled estimand: the bad return is removed before both arms, and each task's two immediate gains are equalized. Retaining the confirmed good return selects `concept_0002` on both base tasks and gives about `2.90e-20` held-out loss; deleting it selects `concept_0001` and gives about `8.31e-4`. The shifted and noisy tasks do not receive this credit because their response conditions do not match. This isolates the scheduler tie-break effect while leaving natural multi-source attribution and long-horizon net benefit open.

The `counterfactual_source_replays` section measures each frozen source alone on the same paid prefix and held-out response. It gives a source-conditioned quality readout without calling it a scheduler causal effect. There are zero external model calls.
