# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- loaded shard payloads: `4/4`
- sample count: `4`
- distinct sample problems: `4`
- duplicate sample problems: `0`
- live attempts resolved: `12/12`
- scored rows: `12`
- overall accuracy: `0.0`
- top-level live errors: `0`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `[]`
- pollution failed gates: `[]`
- recommended HLE claim scope: `full_resolved_rows`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `4` | `0.0` | `0` | `0.0` | `0.0` |
| `gpt-5.4-mini` | `hipporag_baseline` | `4` | `0.0` | `0` | `0.0` | `0.0` |
| `gpt-5.4-mini` | `raw` | `4` | `0.0` | `0` | `0.0` | `0.0` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `4` | `0.0` |
| `gpt-5.4-mini` | `hipporag_baseline` | `4` | `0.0` |
| `gpt-5.4-mini` | `raw` | `4` | `0.0` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `jsonl_error_events_by_event` | `recursive_child_error` | `4` |
| `jsonl_error_events_by_variant` | `assumption_agent_recursive_verify` | `4` |
| `jsonl_error_events_by_type` | `RuntimeError` | `4` |
| `process_status_counts` | `completed` | `4` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `4` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `0` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `0` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `4` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `12` |
| `cache_live_separation` | `planned_live_model_calls` | `12` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `12` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `28` |
| `context_pollution_summary` | `evidence_context_used` | `4` |
| `context_pollution_summary` | `evidence_context_wrong` | `4` |
| `context_pollution_summary` | `graph_context_used` | `4` |
| `context_pollution_summary` | `graph_context_wrong` | `4` |
| `context_pollution_summary` | `graph_generic_harness_context` | `4` |
| `context_pollution_summary` | `hipporag_no_results` | `1` |
| `context_pollution_summary` | `morphism_hit` | `2` |
| `context_pollution_summary` | `morphism_wrong` | `2` |
| `pollution_gate` | `cache_live_separation_accounted` | `True` |
| `pollution_gate` | `claim_scope_downgraded_when_endpoint_errors` | `True` |
| `pollution_gate` | `clean_shared_subset_available_if_endpoint_errors` | `True` |
| `pollution_gate` | `context_pollution_accounted` | `True` |
| `pollution_gate` | `endpoint_errors_separated` | `True` |
| `pollution_gate` | `fresh_problem_hashes_accounted` | `True` |
| `pollution_gate` | `no_duplicate_problem_hashes` | `True` |
| `pollution_gate` | `raw_content_not_persisted` | `True` |
| `pollution_gate` | `selection_credit_accounted` | `True` |

## Selection Credit

| method | n | correct | error | accuracy |
| --- | ---: | ---: | ---: | ---: |
| `math_exact_direct_fallback` | `1` | `0` | `0` | `0.0` |
| `normalized_majority` | `3` | `0` | `0` | `0.0` |
| `unknown` | `8` | `0` | `0` | `0.0` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `736.9808` | `1` | `2200` | `none` |
| `1` | `completed` | `0` | `257.8939` | `1` | `2223` | `none` |
| `2` | `completed` | `0` | `1989.486` | `1` | `2246` | `none` |
| `3` | `completed` | `0` | `1323.0471` | `1` | `2269` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
