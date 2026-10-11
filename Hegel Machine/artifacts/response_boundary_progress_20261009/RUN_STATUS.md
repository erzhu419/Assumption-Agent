# Target support and experiment value — 2026-10-09

Reused 0.112 parameter-domain contraction, 0.15 boundary-aware action selection, and the existing Boolean response solver. The proposed diagnosis of missed reference-boundary progress was false: the original conditional target support was 250 → 246 → 246 → 246 settings. Both reference negatives, and even all six neighbouring reference negatives together, leave the 246-setting support unchanged; see `DIAGNOSIS.md` and `full_grid.json`.

The solver now projects attainable public-grid settings for bounded scalar supports. Both possible responses are compared before choosing an operational target trial; this reduction is separate from nonlocal explanatory progress. Stalled research remains stopped, with the remaining support and a suggested trial passed to the existing parent research obligation. Impossible target gaps are removed before charging the existing branch limit.

| Exact-prefix replay | Before: actions / research compute | Current: actions / research compute | Completion |
|---|---:|---:|---|
| task01, 160 observations | 25 / 385,395 | 32 / 399,659 | Neither completes |
| task02, 120 observations | 31 / 29,695 | 31 / 29,695 | Both complete |

Final task01 support is 237 settings, returned with the conditional assumption to the parent. Source: `results.json`, `success_control/results.json`, and `boundary_progress/analysis.json`. Projection and nested-research regressions: 21 passed. All runs reuse recorded proposals; zero new LLM calls.

## Limitations and next step

The first automatic-search variant failed after 90 actions (`unpruned/results.json`). After fixing branch pruning, continued automatic search still enumerated points; it was intentionally interrupted after at least 159 actions, with no final verdict (`continued_search/INTERRUPTED.json`). Neither variant is the final policy. A negative membership observation does not eliminate half an interval. The final paired replay also has no success gain and adds seven actions on the failure. Projection covers bounded scalar supports; other cases remain unresolved. Next, test whether the existing LLM research loop uses this explicit ambiguity to obtain a more informative measurement or revise its assumptions. These opened conditions establish neither autonomous adoption nor new-task gains.
