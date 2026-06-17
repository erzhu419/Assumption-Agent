# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- loaded shard payloads: `1/1`
- sample count: `1`
- distinct sample problems: `1`
- duplicate sample problems: `0`
- live attempts resolved: `2/2`
- scored rows: `2`
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
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `1` | `0.0` | `0` | `0.0` | `None` |
| `gpt-5.4-mini` | `raw` | `1` | `0.0` | `0` | `0.0` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `1` | `0.0` |
| `gpt-5.4-mini` | `raw` | `1` | `0.0` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `process_status_counts` | `completed` | `1` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `1` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `0` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `0` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `1` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `2` |
| `cache_live_separation` | `planned_live_model_calls` | `2` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `2` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `5` |
| `context_pollution_summary` | `generic_graph_context_only` | `1` |
| `context_pollution_summary` | `graph_context_discarded` | `1` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `1` |
| `context_pollution_summary` | `graph_retrieval_activated` | `1` |
| `context_pollution_summary` | `morphism_hit` | `1` |
| `context_pollution_summary` | `morphism_wrong` | `1` |
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
| `unknown` | `1` | `0` | `0` | `0.0` |
| `verified_or_abstain_direct_fallback` | `1` | `0` | `0` | `0.0` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `1` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `1` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `1` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `1` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `1` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `1` |
| `verified_or_abstain_gate_status` | `abstained` | `1` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `1` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `1` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `224.7309` | `1` | `1843` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
