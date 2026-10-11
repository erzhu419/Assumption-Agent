# Shared pair confirmation — completed negative result

Frozen 0.174: 78 relevant tests passed; 36 learning chains completed before all 504 terminal scores. Offline, zero model API calls.
Each of 3 arms used the same 12 roots ×14 tasks, task budgets and terminal seeds; old 0.173 and 0.164 results were reused.
Full / no-cost: 135/168 solved, MSE 0.03339847; noqueue: 137/168, 0.02112040. Later seven: 69/84 versus 70/84.
Full spent 89,265 observations /87,657 measurements /46,806,195 compute, versus 87,249 /85,641 /44,811,676; CPU +40.87%.
The noqueue control exactly reproduced all 168 old task results and all 12 resource ledgers.
163 independent pair visits bought 3,912 rows; 10 batches distinguished the explanations (5 positive and 5 negative gains); one old explanation was restored.
Cost feedback changed 376 expected-cost quotations, but no resource share, priority, visit sequence, quality or total resource outcome.
The global queue candidate is rejected; the project returns to the reproduced 0.173 reference.
Details: `execution_summary.json`, `compact_results.json`, `paired_details.json`, `root10_pair_queue_diagnosis.json`.

Limitations: the reachable stale-evidence bug had zero instances in this cohort. Its separate 0.174.1 correction stopped with 2/24 learning chains complete and no terminal scoring; frozen source, tests and partial outputs remain in `correction/`.
