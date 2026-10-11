# Condition revision utility in current-risk units — 0.175.1

Complete and negative; active source restored to0.173. Frozen candidate source and all results remain.
39 candidate tests passed. All36 learning chains completed before504 fixed terminal scores; zero external API calls.
Seen roots01–12: candidate128/168, MSE0.0468654; reused0.173137/168,0.0211204. Last7:67/84 versus70/84.
New roots13–24: candidate138/168, MSE0.0366794; newly run0.173147/168,0.0352685.
The new cohort's first7 lost9 passes (65/84 versus74/84). Last7 passed73/84 in both, with MSE0.0157612 versus0.0205933.
Seen failures/recoveries18/9, new16/7; task W/T/L12/135/21 and9/141/18.14 of30 prior0.175 regressions were repaired.
All observation/measurement/compute: seen79680/78072/42267586 versus87249/85641/44811676; new80296/78856/40852391 versus89076/87612/44973008.
CPU: seen2175.35 versus3883.10seconds; new2243.49 versus3508.15seconds. Actual revisits:56 versus190 and59 versus238.
`compact_results.json`, `paired_details.json` and `execution_summary.json` retain every failure, recovery and recorded procurement decision.
Candidate test evidence is `tests_unique.json`; restoration validation passed 14 tests (`restored_173_tests.xml`) and does not increase the candidate test count.

Limitations: seen roots were previously evaluated and their reference CPU is from a different run. New roots are consecutive new instances in the same task family. Final screen caches do not count all historical pricing requests. Savings and the new cohort's last-seven MSE improvement do not remove the loss of overall accuracy.
