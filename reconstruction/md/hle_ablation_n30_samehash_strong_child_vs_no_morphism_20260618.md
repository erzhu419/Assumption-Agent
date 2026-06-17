# HLE Paired Run Comparison

- eval id: `hle_ablation_n30_samehash_strong_child_vs_no_morphism_20260618`
- pass: `True`
- failed gates: `[]`
- candidate variant: `assumption_agent_recursive_verify`
- candidate problem count: `30`
- candidate accuracy: `0.5333333333333333`
- baseline profile: `no_morphism`
- raw content persisted: `False`

## Paired Deltas

| baseline variant | n | candidate acc | baseline acc | delta | wins | losses | p | ci95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `assumption_agent_recursive_verify` | `30` | `0.5333333333333333` | `0.26666666666666666` | `0.26666666666666666` | `10` | `2` | `0.03857421875` | `[0.06666666666666667, 0.4666666666666667]` |

## Oracle Pattern

- shared n: `30`
- baseline order: `['assumption_agent_recursive_verify']`
- candidate-only correct: `10`
- all wrong: `12`
- oracle accuracy: `0.6`

| pattern | count |
| --- | ---: |
| `00` | `12` |
| `01` | `10` |
| `10` | `2` |
| `11` | `6` |

## Validation Gates

| gate | pass |
| --- | ---: |
| `candidate_error_free` | `True` |
| `candidate_not_below_primary` | `True` |
| `candidate_rows_present` | `True` |
| `expected_sample_complete` | `True` |
| `primary_shared_n_positive` | `True` |
| `raw_content_not_persisted` | `True` |

The comparison uses hashes and correctness booleans only; raw HLE content is not persisted.
