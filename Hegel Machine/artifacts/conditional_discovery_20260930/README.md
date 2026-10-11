# Conditional and temporal discovery, 2026-09-30

The 0.133 baseline and 0.134 candidate are frozen in `baseline/src` and each cohort’s `runtime/src`. All runs use offline search, generic public controls and a private noiseless apparatus. Per task: 240 observations, 240,000 compute units; cumulative revisits use unused visited-task resources.

Both cohorts finished all research before common 4 × 64 terminal scoring: development 42 tasks, independent formal replication 252 tasks across 12 new apparatus (9914701–9914712). No task or negative control was excluded. The full results, stage breakdown and actual executed structures are in each `summary.json`.

| Formal arm | MSE | Solved /84 | False completed | Research CPU | Observations | Compute units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old_cumulative | 5.289024 | 0 | 0 | 434.65s | 18637 | 13228224 |
| new_cumulative | 2.233605 | 27 | 11 | 2069.56s | 16698 | 16812412 |
| new_independent | 4.702962 | 26 | 9 | 521.07s | 14922 | 12248953 |

New cumulative MSE is 57.77% below old cumulative and 52.51% below new independent; research CPU is 4.76× and 3.97× respectively. Each new arm solves all 24 plain/no-branch window controls. Cumulative solves only 3/60 branching tasks; historical branches, changed window/boundary and renamed/scaled stages each remain 0/12 solved. Development likewise lowered MSE 67.16% with 6.77× research CPU and 2 false completions.

Limitations: generalized branch discovery, boundary coverage and long-term net benefit remain open. Independent resets knowledge and unused budget, so it is not a vanilla Code/LLM baseline. Retrospective development diagnosis found successor adoption left a stale current research graph; its stored discovery-step metadata can include legitimate history, so actual executed structures are reported separately. No frozen candidate was altered or formal run retuned.
