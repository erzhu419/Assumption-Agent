# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- process timeout policy: `watch_only`
- kill on soft timeout: `False`
- shard sample dedupe: `completed`
- loaded shard payloads: `30/30`
- sample count: `30`
- distinct sample problems: `30`
- duplicate sample problems: `0`
- live attempts resolved: `30/30`
- scored rows: `30`
- overall accuracy: `0.5333`
- top-level live errors: `0`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `[]`
- pollution failed gates: `[]`
- recommended HLE claim scope: `full_resolved_rows`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `30` | `0.5333` | `0` | `0.5333` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `30` | `0.5333` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `jsonl_error_events_by_event` | `recursive_child_error` | `8` |
| `jsonl_error_events_by_variant` | `assumption_agent_recursive_verify` | `8` |
| `jsonl_error_events_by_type` | `RuntimeError` | `8` |
| `jsonl_error_events_by_label` | `model request failed: URLError_SSLEOFError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violat...` | `8` |
| `process_status_counts` | `completed` | `30` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `30` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `0` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `0` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `30` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `30` |
| `cache_live_separation` | `planned_live_model_calls` | `30` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `30` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `170` |
| `context_pollution_summary` | `generic_graph_context_only` | `26` |
| `context_pollution_summary` | `graph_context_correct` | `2` |
| `context_pollution_summary` | `graph_context_discarded` | `26` |
| `context_pollution_summary` | `graph_context_used` | `2` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `26` |
| `context_pollution_summary` | `graph_retrieval_activated` | `30` |
| `context_pollution_summary` | `hipporag_context_correct` | `3` |
| `context_pollution_summary` | `hipporag_context_used` | `5` |
| `context_pollution_summary` | `hipporag_context_wrong` | `2` |
| `context_pollution_summary` | `hipporag_no_results` | `1` |
| `context_pollution_summary` | `morphism_correct` | `8` |
| `context_pollution_summary` | `morphism_hit` | `12` |
| `context_pollution_summary` | `morphism_wrong` | `4` |
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
| `verified_or_abstain_direct_fallback` | `29` | `15` | `0` | `0.5172` |
| `verifier_choice` | `1` | `1` | `0` | `1.0` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `14` |
| `agent_failure_buckets` | `hipporag_context_invalid_or_unhelpful` | `2` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `14` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `14` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `4` |
| `agent_gain_loss` | `agent_correct_raw_wrong` | `16` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `29` |
| `agent_selection_methods` | `verifier_choice` | `1` |
| `verified_or_abstain_gate_status` | `abstained` | `29` |
| `verified_or_abstain_gate_status` | `allowed` | `1` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::correct` | `16` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `14` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::correct` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::wrong_or_error` | `4` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::correct` | `14` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `8` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `None` | `1` | `0` | `none` |
| `1` | `completed` | `0` | `None` | `1` | `1` | `none` |
| `2` | `completed` | `0` | `None` | `1` | `2` | `none` |
| `3` | `completed` | `0` | `None` | `1` | `3` | `none` |
| `4` | `completed` | `0` | `None` | `1` | `4` | `none` |
| `5` | `completed` | `0` | `None` | `1` | `5` | `none` |
| `6` | `completed` | `0` | `None` | `1` | `6` | `none` |
| `7` | `completed` | `0` | `None` | `1` | `7` | `none` |
| `8` | `completed` | `0` | `None` | `1` | `8` | `none` |
| `9` | `completed` | `0` | `None` | `1` | `9` | `none` |
| `10` | `completed` | `0` | `None` | `1` | `10` | `none` |
| `11` | `completed` | `0` | `None` | `1` | `11` | `none` |
| `12` | `completed` | `0` | `None` | `1` | `12` | `none` |
| `13` | `completed` | `0` | `None` | `1` | `13` | `none` |
| `14` | `completed` | `0` | `None` | `1` | `14` | `none` |
| `15` | `completed` | `0` | `None` | `1` | `15` | `none` |
| `16` | `completed` | `0` | `None` | `1` | `16` | `none` |
| `17` | `completed` | `0` | `None` | `1` | `17` | `none` |
| `18` | `completed` | `0` | `None` | `1` | `18` | `none` |
| `19` | `completed` | `0` | `None` | `1` | `19` | `none` |
| `20` | `completed` | `0` | `None` | `1` | `20` | `none` |
| `21` | `completed` | `0` | `None` | `1` | `21` | `none` |
| `22` | `completed` | `0` | `None` | `1` | `22` | `none` |
| `23` | `completed` | `0` | `None` | `1` | `23` | `none` |
| `24` | `completed` | `0` | `None` | `1` | `24` | `none` |
| `25` | `completed` | `0` | `None` | `1` | `25` | `none` |
| `26` | `completed` | `0` | `None` | `1` | `26` | `none` |
| `27` | `completed` | `0` | `None` | `1` | `27` | `none` |
| `28` | `completed` | `0` | `None` | `1` | `28` | `none` |
| `29` | `completed` | `0` | `None` | `1` | `29` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
