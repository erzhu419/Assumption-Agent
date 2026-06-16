# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `False`
- loaded shard payloads: `10/10`
- sample count: `30`
- distinct sample problems: `30`
- duplicate sample problems: `0`
- live attempts resolved: `90/90`
- scored rows: `90`
- overall accuracy: `0.1444`
- top-level live errors: `22`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `['zero_top_level_live_errors']`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `30` | `0.2` | `1` | `0.3333` | `0.1429` |
| `gpt-5.4-mini` | `hipporag_baseline` | `30` | `0.1333` | `11` | `0.2222` | `0.0952` |
| `gpt-5.4-mini` | `raw` | `30` | `0.1` | `10` | `0.1111` | `0.0952` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `19` | `0.2632` |
| `gpt-5.4-mini` | `hipporag_baseline` | `19` | `0.2105` |
| `gpt-5.4-mini` | `raw` | `19` | `0.1579` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `top_level_errors_by_variant` | `assumption_agent_recursive_verify` | `1` |
| `top_level_errors_by_variant` | `hipporag_baseline` | `11` |
| `top_level_errors_by_variant` | `raw` | `10` |
| `top_level_errors_by_type` | `RuntimeError` | `22` |
| `top_level_errors_by_variant_type` | `assumption_agent_recursive_verify::RuntimeError` | `1` |
| `top_level_errors_by_variant_type` | `hipporag_baseline::RuntimeError` | `11` |
| `top_level_errors_by_variant_type` | `raw::RuntimeError` | `10` |
| `jsonl_error_events_by_event` | `call_error` | `22` |
| `jsonl_error_events_by_event` | `math_tool_child_error` | `5` |
| `jsonl_error_events_by_event` | `recursive_child_error` | `43` |
| `jsonl_error_events_by_variant` | `assumption_agent_recursive_verify` | `49` |
| `jsonl_error_events_by_variant` | `hipporag_baseline` | `11` |
| `jsonl_error_events_by_variant` | `raw` | `10` |
| `jsonl_error_events_by_type` | `RuntimeError` | `70` |
| `process_status_counts` | `completed` | `10` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `1443.6946` | `3` | `1600` | `none` |
| `1` | `completed` | `0` | `1792.7793` | `3` | `1640` | `none` |
| `2` | `completed` | `0` | `805.8677` | `3` | `1680` | `none` |
| `3` | `completed` | `0` | `912.1378` | `3` | `1720` | `none` |
| `4` | `completed` | `0` | `671.5067` | `3` | `1760` | `none` |
| `5` | `completed` | `0` | `2368.9161` | `3` | `1800` | `none` |
| `6` | `completed` | `0` | `1599.1039` | `3` | `1840` | `none` |
| `7` | `completed` | `0` | `1127.6143` | `3` | `1880` | `none` |
| `8` | `completed` | `0` | `1057.496` | `3` | `1920` | `none` |
| `9` | `completed` | `0` | `951.3087` | `3` | `1960` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
