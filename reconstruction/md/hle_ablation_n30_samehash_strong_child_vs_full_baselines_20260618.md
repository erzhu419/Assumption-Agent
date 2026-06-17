# HLE Paired Run Comparison

- eval id: `hle_ablation_n30_samehash_strong_child_vs_full_baselines_20260618`
- pass: `True`
- failed gates: `[]`
- candidate variant: `assumption_agent_recursive_verify`
- candidate problem count: `30`
- candidate accuracy: `0.5333333333333333`
- baseline profile: `full`
- raw content persisted: `False`

## Paired Deltas

| baseline variant | n | candidate acc | baseline acc | delta | wins | losses | p | ci95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `assumption_agent_recursive_verify` | `30` | `0.5333333333333333` | `0.26666666666666666` | `0.26666666666666666` | `9` | `1` | `0.021484375` | `[0.1, 0.4666666666666667]` |
| `hipporag_baseline` | `30` | `0.5333333333333333` | `0.2` | `0.3333333333333333` | `11` | `1` | `0.00634765625` | `[0.13333333333333333, 0.5333333333333333]` |
| `raw` | `30` | `0.5333333333333333` | `0.23333333333333334` | `0.3` | `9` | `0` | `0.00390625` | `[0.13333333333333333, 0.4666666666666667]` |

## Oracle Pattern

- shared n: `30`
- baseline order: `['raw', 'assumption_agent_recursive_verify', 'hipporag_baseline']`
- candidate-only correct: `5`
- all wrong: `13`
- oracle accuracy: `0.5666666666666667`

| pattern | count |
| --- | ---: |
| `0000` | `13` |
| `0001` | `5` |
| `0011` | `1` |
| `0101` | `2` |
| `0110` | `1` |
| `0111` | `1` |
| `1001` | `1` |
| `1011` | `2` |
| `1101` | `3` |
| `1111` | `1` |

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
