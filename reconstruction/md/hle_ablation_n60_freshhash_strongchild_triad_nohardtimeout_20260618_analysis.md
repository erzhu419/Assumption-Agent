# HLE Ablation Result Analysis

- eval id: `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618`
- profile count: `1`
- pass: `True`
- failed gates: `[]`
- raw content persisted: `False`

## Profile Accuracy

| profile | clean | variant | n | accuracy | errors |
| --- | ---: | --- | ---: | ---: | ---: |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `True` | `assumption_agent_recursive_verify` | `60` | `0.4` | `0` |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `True` | `hipporag_baseline` | `60` | `0.16666666666666666` | `0` |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `True` | `raw` | `60` | `0.13333333333333333` | `0` |

## Same-Profile Paired Deltas

| profile | pair | n | delta | wins | losses | p | ci95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `agent_vs_hipporag` | `60` | `0.23333333333333334` | `17` | `3` | `0.0025768280029296875` | `[0.1, 0.36666666666666664]` |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `agent_vs_raw` | `60` | `0.26666666666666666` | `18` | `2` | `0.0004024505615234375` | `[0.13333333333333333, 0.4]` |

## Pollution

- contaminated profiles: `[]`

| profile | contaminated | disabled-module activations | raw content persisted |
| --- | ---: | --- | ---: |
| `hle_ablation_n60_freshhash_strongchild_triad_nohardtimeout_20260618` | `False` | `{}` | `False` |

The analysis uses hashes and correctness booleans only; raw HLE content is not persisted.
