# Competing measured-condition research, 0.139

Reuses `scripts/run_observed_condition_decoding_study.py` and the real uncontrollable-instrument laboratory. Development: two retained apparatus, 42 task-arm outcomes. Formal: 12 fresh apparatus (9915701–12), 252 task-arm outcomes. All first visits keep 320 observations, 320 measurement units and 320000 computation; strict terminal success is MSE ≤ 1e-6 on four common 64-row trajectories.

Old cumulative is frozen 0.138. New cumulative and independent begin after root confirms the candidate source is ready and frozen. Independent resets the entire Hegel library each task; it is not a vanilla Code/LLM comparator. Development baseline reuses learning only. All learning and natural revisits finish before unique common terminal evaluation. Offline search; zero external model calls.

Development learning and common terminal scoring are complete: old cumulative 9/14 solved, 4 false completions; new cumulative 9/14 solved, 0 false completions; new independent 9/14 solved, 0 false completions. Both new arms report 10/14 tasks unresolved. New cumulative mean response MSE is 0.071924 versus old 0.090682, with CPU 77.60 s versus 44.93 s and observations 3600 versus 2112. Lower false completion alone is not an accuracy gain. Renamed and returning conditional tasks retain negative transfer outcomes.

Formal learning, natural revisits and unique common terminal evaluation are complete (252 task-arm outcomes). Old 0.138 cumulative: 51/84 solved, 32 false completions, 83 completed and one unresolved. New 0.139 cumulative: 53/84 solved, zero false completions, 24 completed and 60 unresolved. New 0.139 independent: 54/84 solved, zero false completions, 24 completed and 60 unresolved. All 60 conditional tasks in each new arm remain unresolved.

Formal cumulative mean response MSE is 0.351262 versus old 0.386377 (−9.09%) and independent 0.402611. Cumulative learning CPU is 556.51 s versus old 403.67 s (+37.86%) and independent 463.25 s. Old→new cumulative observations: 13835→22696; measurement units: 11963→20256; computation: 10897298→12334953. The cumulative arm runs 372 competition batches. Root05 renamed/scaled W7 regresses from old MSE 3.67e-18 to 14.952; returning and historical tasks retain other negative transfers. These outcomes do not establish stable net knowledge benefit. Full positive and negative outcomes are in `fresh_replication/summary.json`.

Direct source tests: 42 passed (root-owned test run). Development stores partial research drafts; preserved drafts alone do not establish later replay or knowledge gains.

Development research consumption: 68 competition batches bought 1632 rows, narrowed intervals in 19 batches and changed the fitted response in three cases (original measured losses 1.119/1.266/10.154 to below 1e-12). Forty-nine batches neither narrowed the interval nor revised the response, costing 1176 rows and 149090 computation. Ten partial drafts were preserved; actual later replay, evaluation and adoption were all zero.

草稿实际采用补充：[正式消费记录](fresh_replication/knowledge_consumption.json)确认50份数值草稿、12次回放产生5个候选，验证MSE为1.83–4.19，均为pending；没有采用后继关系或已解决草稿。开发无回放、正式少量尝试未形成迁移收益，二者分别记录。
