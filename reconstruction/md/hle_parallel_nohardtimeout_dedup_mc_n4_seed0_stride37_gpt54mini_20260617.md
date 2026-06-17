# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- process timeout policy: `watch_only`
- kill on soft timeout: `False`
- shard sample dedupe: `completed`
- loaded shard payloads: `4/4`
- sample count: `4`
- distinct sample problems: `4`
- duplicate sample problems: `0`
- live attempts resolved: `8/8`
- scored rows: `8`
- overall accuracy: `0.375`
- top-level live errors: `0`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `[]`
- pollution failed gates: `[]`
- recommended HLE claim scope: `full_resolved_rows`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `4` | `0.5` | `0` | `0.5` | `None` |
| `gpt-5.4-mini` | `raw` | `4` | `0.25` | `0` | `0.25` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `4` | `0.5` |
| `gpt-5.4-mini` | `raw` | `4` | `0.25` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `process_status_counts` | `completed` | `4` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `4` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `4` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `364` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `4` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `8` |
| `cache_live_separation` | `planned_live_model_calls` | `8` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `8` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `28` |
| `context_pollution_summary` | `generic_graph_context_only` | `2` |
| `context_pollution_summary` | `graph_context_discarded` | `2` |
| `context_pollution_summary` | `graph_context_used` | `1` |
| `context_pollution_summary` | `graph_context_wrong` | `1` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `2` |
| `context_pollution_summary` | `graph_retrieval_activated` | `4` |
| `context_pollution_summary` | `hipporag_context_correct` | `1` |
| `context_pollution_summary` | `hipporag_context_used` | `1` |
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
| `unknown` | `4` | `1` | `0` | `0.25` |
| `verified_or_abstain_direct_fallback` | `4` | `2` | `0` | `0.5` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `2` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `2` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `2` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `1` |
| `agent_failure_buckets` | `weak_morphism_unhelpful` | `1` |
| `agent_gain_loss` | `agent_correct_raw_wrong` | `1` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `2` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `4` |
| `verified_or_abstain_gate_status` | `abstained` | `4` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::correct` | `2` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `2` |
| `by_variant_answer_type::raw` | `multipleChoice::correct` | `1` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `3` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::correct` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::correct` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `2` |
| `by_variant_domain::raw` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `science::correct` | `1` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `2` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `415.1005` | `1` | `0` | `none` |
| `1` | `completed` | `0` | `168.5093` | `1` | `74` | `none` |
| `2` | `completed` | `0` | `188.4901` | `1` | `111` | `none` |
| `3` | `completed` | `0` | `336.5974` | `1` | `148` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
