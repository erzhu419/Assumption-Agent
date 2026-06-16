# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `False`
- loaded shard payloads: `4/4`
- sample count: `12`
- distinct sample problems: `12`
- duplicate sample problems: `0`
- live attempts resolved: `36/36`
- scored rows: `36`
- overall accuracy: `0.1667`
- top-level live errors: `2`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `['zero_top_level_live_errors']`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `12` | `0.25` | `0` | `0.3333` | `0.2222` |
| `gpt-5.4-mini` | `hipporag_baseline` | `12` | `0.0833` | `1` | `0.0` | `0.1111` |
| `gpt-5.4-mini` | `raw` | `12` | `0.1667` | `1` | `0.3333` | `0.1111` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `10` | `0.2` |
| `gpt-5.4-mini` | `hipporag_baseline` | `10` | `0.1` |
| `gpt-5.4-mini` | `raw` | `10` | `0.1` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `top_level_errors_by_variant` | `hipporag_baseline` | `1` |
| `top_level_errors_by_variant` | `raw` | `1` |
| `top_level_errors_by_type` | `RuntimeError` | `2` |
| `top_level_errors_by_variant_type` | `hipporag_baseline::RuntimeError` | `1` |
| `top_level_errors_by_variant_type` | `raw::RuntimeError` | `1` |
| `jsonl_error_events_by_event` | `call_error` | `2` |
| `jsonl_error_events_by_event` | `recursive_child_error` | `5` |
| `jsonl_error_events_by_variant` | `assumption_agent_recursive_verify` | `5` |
| `jsonl_error_events_by_variant` | `hipporag_baseline` | `1` |
| `jsonl_error_events_by_variant` | `raw` | `1` |
| `jsonl_error_events_by_type` | `RuntimeError` | `7` |
| `process_status_counts` | `completed` | `4` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `2526.6805` | `3` | `1600` | `none` |
| `1` | `completed` | `0` | `2011.445` | `3` | `1640` | `none` |
| `2` | `completed` | `0` | `991.6813` | `3` | `1680` | `none` |
| `3` | `completed` | `0` | `1113.8869` | `3` | `1720` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
