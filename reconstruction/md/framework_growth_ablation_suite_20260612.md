# Framework Growth Ablation Suite

- pass: `True`
- full score: `0.7668`
- best toggle-off score: `0.6064`
- margin vs best toggle-off: `0.1604`

| Variant | Score | Old Preservation | Residual | Limiting | Generality | New Prediction | Regression | Readback | Unsafe Promotions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_dialectical_framework_growth` | `0.7668` | `1.0` | `0.79` | `0.9533` | `0.3749` | `0.8067` | `0.0` | `1.0` | `0` |
| `no_conservative_gate_residual_only` | `0.2844` | `0.83` | `0.82` | `0.36` | `0.18` | `0.58` | `0.11` | `0.42` | `2` |
| `no_branch_ledger_no_pruning` | `0.4321` | `0.88` | `0.77` | `0.74` | `0.25` | `0.66` | `0.08` | `0.63` | `1` |
| `no_graph_lifecycle_score_only` | `0.6064` | `0.96` | `0.78` | `0.91` | `0.31` | `0.72` | `0.02` | `0.0` | `0` |
| `no_limiting_case_gate` | `0.4831` | `0.89` | `0.81` | `0.25` | `0.22` | `0.62` | `0.09` | `0.5` | `1` |
| `no_old_success_preservation_gate` | `0.431` | `0.79` | `0.84` | `0.62` | `0.28` | `0.7` | `0.14` | `0.58` | `2` |
| `local_patch` | `0.3605` | `0.88` | `0.69` | `0.44` | `0.11` | `0.5` | `0.075` | `0.25` | `1` |
| `raw_wisdom` | `0.1285` | `0.79` | `0.57` | `0.22` | `0.05` | `0.38` | `0.14` | `0.1` | `2` |
