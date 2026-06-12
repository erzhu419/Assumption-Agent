# Conservative Generalization Gate

- pass: `True`
- candidates: `4`
- decisions: `{'active_scoped_framework': 1, 'branch_only': 1, 'candidate_framework': 1, 'reject': 1}`
- active required relation coverage: `1.0`
- top framework growth score: `0.7881`

## Evaluation Rows

| Framework | Decision | Growth | Old Preservation | Residual Explanation | Limiting Reduction | Generality | New Prediction | Regression |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fw_dependency_aware_controlled_intervention` | `active_scoped_framework` | `0.7881` | `1.0` | `0.79` | `0.9533` | `0.3749` | `0.8067` | `0.0` |
| `fw_evidence_ladder_verifier_routing` | `candidate_framework` | `0.7222` | `1.0` | `0.73` | `0.9` | `0.2406` | `0.74` | `0.0` |
| `fw_boundary_first_analogy_abstention` | `branch_only` | `0.6673` | `1.0` | `0.675` | `0.88` | `0.1076` | `0.66` | `0.0` |
| `fw_longer_context_style_boost` | `reject` | `0.4639` | `0.902` | `0.57` | `0.58` | `0.0` | `0.54` | `0.0933` |

## Required Relations

- `fw_dependency_aware_controlled_intervention`: `['conflicts_with', 'explains_residual', 'generalizes', 'modifies_boundary_of', 'predicts_new_case', 'preserves_success_cases', 'reduces_to_under_scope']`
- `fw_evidence_ladder_verifier_routing`: `['conflicts_with', 'explains_residual', 'generalizes', 'modifies_boundary_of', 'predicts_new_case', 'preserves_success_cases', 'reduces_to_under_scope']`

## Claim Boundary

This is a bounded framework-growth gate.  It does not claim unbounded philosophy generation,
ungated framework promotion, replacement of live validation, or a full theorem prover.
