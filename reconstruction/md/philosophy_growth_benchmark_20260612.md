# Philosophy Growth Benchmark

- pass: `True`
- framework growth score: `0.8361`
- conservative vs local patch margin: `0.2567`
- active framework survival count: `5`
- core prior promotions: `0`

## Baselines

| Policy | Growth | Old Preservation | Residual Explanation | Limiting Reduction | Generality | New Prediction | Regression |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `conservative_generalization` | `0.7478` | `1.0` | `0.79` | `0.9533` | `0.3749` | `0.8067` | `0.0` |
| `local_patch` | `0.4911` | `0.88` | `0.69` | `0.44` | `0.11` | `0.5` | `0.075` |
| `raw_wisdom` | `0.3546` | `0.79` | `0.57` | `0.22` | `0.05` | `0.38` | `0.14` |

## Evolution

- generation `1`: status `candidate_framework`, score `0.7222`, action `retain_for_more_validation`
- generation `2`: status `active_scoped_framework`, score `0.7881`, action `promote_scoped_after_old_success_and_residual_tests`
- generation `3`: status `active_scoped_framework`, score `0.8001`, action `survival_recheck_unseen_domain`
- generation `4`: status `active_scoped_framework`, score `0.8121`, action `prune_failed_prompt_style_branch`
- generation `5`: status `active_scoped_framework`, score `0.8241`, action `monitor_descendant_productivity_without_core_promotion`
- generation `6`: status `active_scoped_framework`, score `0.8361`, action `monitor_descendant_productivity_without_core_promotion`
- generation `6`: status `rejected_boundary_only`, score `0.4639`, action `retain_negative_evidence_do_not_delete`
