# 0.131 research investment and reuse cost

All tasks use the offline search proposer. No external model calls.

- `baseline/src`: frozen 0.130 runtime, directly equal to the previous formal runtime before copying development reference results.
- `development`: apparatus 9914203/08/11; old cumulative, new cumulative and new independent, 18 tasks per arm. New cumulative improves mean-response MSE by 16.16% and reduces measured research CPU by 24.74%, with 9.37% more observations. The 16→13 completion decline is retained.
- `fresh_replication/protocol.json`: 12 previously unused apparatus 9914301–12, six windows 3/5/7/2/9/return3, three arms and 216 tasks. Same per-task/chain budgets, Gaussian response noise .01, and common post-learning 4×64 terminal trajectories. The candidate runtime is copied exactly from the developed runtime; no formal outcome tunes it.
- `profile_root08`: a complete already-seen 0.130 chain profiled for diagnosis. Profiler timings include instrumentation overhead. `propose_research_programs` accounts for 308.19 of 412.11 profiler seconds, versus 24.29 seconds for successor offers.
- `compare_results.py`, `attribute_cpu.py`, `attribute_research.py`: read completed artifacts to pair outcomes, account for research CPU, and count retained research actions and full-chain offers. They do not call the Agent or acquire training data.

Formal study finished naturally with exit code 0: 36 learning chains, 216 terminal tasks, 864 trajectories and 55,296 measured terminal rows. Every task has four 64-step predictive and mean-response trajectories. No study processes remain.

Old→new cumulative: research CPU 3122.43→885.66s (−71.64%), compute 9,244,527→3,443,115 (−62.76%), observations 14,197→15,909 (+12.06%). Mean-response MSE 1.41771e−5→1.25300e−5 (−11.62%), with 34 wins/11 ties/27 losses and 7 improving/5 worsening apparatus. Both complete 68/72 tasks. Including terminal scoring, CPU is 3127.37→890.33s and observations 32,629→34,341.

Program-expansion actions fall from 67 to 10; 111 paid early noise tests support 66 deferrals. Successor offers fall 161→98, screening compute 1,988,546→639,083. Sources are used by 40/60 successor tasks, versus 30/60 before.

## Limitations

Cumulative learning still has 4.02% higher mean-response MSE and 3.70 times the research CPU of new independent learning; stable net utility from accumulated knowledge is not established. The worst task, apparatus 9914303/window7, rises from 1.13970e−5 to 1.19542e−4 despite the new completion status. Its reports are preserved in each arm's `fresh_replication/runs/*/root03/checkpoint.json.gz`, entry `results[2].research`, alongside `evaluation.json`.

The study covers one controlled response family. Research CPU measures `session.run`, excluding checkpoint writing/reloading; all reported source, research and terminal charges are retained, and profiler overhead is diagnostic only.
