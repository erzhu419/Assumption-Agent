# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `False`
- pollution pass: `True`
- loaded shard payloads: `12/12`
- sample count: `12`
- distinct sample problems: `12`
- duplicate sample problems: `0`
- live attempts resolved: `36/36`
- scored rows: `36`
- overall accuracy: `0.0833`
- top-level live errors: `5`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `['zero_top_level_live_errors']`
- pollution failed gates: `[]`
- recommended HLE claim scope: `clean_shared_subset_due_to_endpoint_noise`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `12` | `0.0833` | `0` | `0.0` | `0.1` |
| `gpt-5.4-mini` | `hipporag_baseline` | `12` | `0.0833` | `3` | `0.0` | `0.1` |
| `gpt-5.4-mini` | `raw` | `12` | `0.0833` | `2` | `0.0` | `0.1` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `8` | `0.125` |
| `gpt-5.4-mini` | `hipporag_baseline` | `8` | `0.125` |
| `gpt-5.4-mini` | `raw` | `8` | `0.125` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `top_level_errors_by_variant` | `hipporag_baseline` | `3` |
| `top_level_errors_by_variant` | `raw` | `2` |
| `top_level_errors_by_type` | `RuntimeError` | `5` |
| `top_level_errors_by_variant_type` | `hipporag_baseline::RuntimeError` | `3` |
| `top_level_errors_by_variant_type` | `raw::RuntimeError` | `2` |
| `jsonl_error_events_by_event` | `call_error` | `5` |
| `jsonl_error_events_by_event` | `recursive_child_error` | `4` |
| `jsonl_error_events_by_variant` | `assumption_agent_recursive_verify` | `4` |
| `jsonl_error_events_by_variant` | `hipporag_baseline` | `3` |
| `jsonl_error_events_by_variant` | `raw` | `2` |
| `jsonl_error_events_by_type` | `RuntimeError` | `9` |
| `process_status_counts` | `completed` | `12` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `12` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `0` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `0` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `12` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `31` |
| `cache_live_separation` | `planned_live_model_calls` | `36` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `36` |
| `cache_live_separation` | `top_level_error_count` | `5` |
| `cache_live_separation` | `underlying_model_calls_executed` | `75` |
| `context_pollution_summary` | `evidence_context_used` | `6` |
| `context_pollution_summary` | `evidence_context_wrong` | `6` |
| `context_pollution_summary` | `generic_graph_context_only` | `12` |
| `context_pollution_summary` | `graph_context_discarded` | `12` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `12` |
| `context_pollution_summary` | `graph_retrieval_activated` | `12` |
| `context_pollution_summary` | `hipporag_context_used` | `2` |
| `context_pollution_summary` | `hipporag_context_wrong` | `2` |
| `context_pollution_summary` | `morphism_hit` | `4` |
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
| `candidate_claim_verifier_priority` | `1` | `1` | `0` | `1.0` |
| `unknown` | `24` | `2` | `5` | `0.0833` |
| `verified_or_abstain_direct_fallback` | `11` | `0` | `0` | `0.0` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `11` |
| `agent_failure_buckets` | `evidence_invalid_or_unhelpful` | `6` |
| `agent_failure_buckets` | `hipporag_context_invalid_or_unhelpful` | `2` |
| `agent_failure_buckets` | `math_exact_failed` | `5` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `2` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `11` |
| `agent_failure_buckets` | `weak_morphism_unhelpful` | `4` |
| `agent_gain_loss` | `all_three_wrong` | `11` |
| `agent_gain_loss` | `hipporag_also_wrong_agent_no_gain` | `11` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `11` |
| `agent_selection_methods` | `candidate_claim_verifier_priority` | `1` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `11` |
| `verified_or_abstain_gate_status` | `abstained` | `11` |
| `verified_or_abstain_gate_status` | `allowed` | `1` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `exactMatch::correct` | `1` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `exactMatch::wrong_or_error` | `9` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `2` |
| `by_variant_answer_type::hipporag_baseline` | `exactMatch::correct` | `1` |
| `by_variant_answer_type::hipporag_baseline` | `exactMatch::wrong_or_error` | `9` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::wrong_or_error` | `2` |
| `by_variant_answer_type::raw` | `exactMatch::correct` | `1` |
| `by_variant_answer_type::raw` | `exactMatch::wrong_or_error` | `9` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::correct` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::wrong_or_error` | `5` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `4` |
| `by_variant_domain::hipporag_baseline` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::hipporag_baseline` | `math::correct` | `1` |
| `by_variant_domain::hipporag_baseline` | `math::wrong_or_error` | `5` |
| `by_variant_domain::hipporag_baseline` | `science::wrong_or_error` | `4` |
| `by_variant_domain::raw` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::raw` | `math::correct` | `1` |
| `by_variant_domain::raw` | `math::wrong_or_error` | `5` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `4` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `968.4526` | `1` | `2300` | `none` |
| `1` | `completed` | `0` | `1806.9591` | `1` | `2317` | `none` |
| `2` | `completed` | `0` | `1016.5866` | `1` | `2334` | `none` |
| `3` | `completed` | `0` | `4211.7327` | `1` | `2351` | `none` |
| `4` | `completed` | `0` | `429.7205` | `1` | `2368` | `none` |
| `5` | `completed` | `0` | `1117.524` | `1` | `2385` | `none` |
| `6` | `completed` | `0` | `514.6696` | `1` | `2402` | `none` |
| `7` | `completed` | `0` | `3002.0769` | `1` | `2419` | `none` |
| `8` | `completed` | `0` | `690.201` | `1` | `2436` | `none` |
| `9` | `completed` | `0` | `872.7653` | `1` | `2453` | `none` |
| `10` | `completed` | `0` | `3333.2183` | `1` | `2470` | `none` |
| `11` | `completed` | `0` | `2522.8277` | `1` | `2487` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
