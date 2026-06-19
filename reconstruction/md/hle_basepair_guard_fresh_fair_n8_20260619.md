# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- parallel workers: `8`
- launch stagger sec: `0.1`
- process timeout policy: `watch_only`
- kill on soft timeout: `False`
- reused completed shards: `0`
- shard sample dedupe: `ok`
- loaded shard payloads: `8/8`
- sample count: `8`
- distinct sample problems: `8`
- duplicate sample problems: `0`
- live attempts resolved: `40/40`
- scored rows: `40`
- overall accuracy: `0.375`
- top-level live errors: `0`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `[]`
- pollution failed gates: `[]`
- model-budget fairness failed gates: `[]`
- recommended HLE claim scope: `full_resolved_rows`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `8` | `0.5` | `0` | `0.5` | `None` |
| `gpt-5.4-mini` | `hipporag_baseline` | `8` | `0.5` | `0` | `0.5` | `None` |
| `gpt-5.4-mini` | `hipporag_budget_matched` | `8` | `0.25` | `0` | `0.25` | `None` |
| `gpt-5.4-mini` | `raw` | `8` | `0.25` | `0` | `0.25` | `None` |
| `gpt-5.4-mini` | `raw_budget_matched` | `8` | `0.375` | `0` | `0.375` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `8` | `0.5` |
| `gpt-5.4-mini` | `hipporag_baseline` | `8` | `0.5` |
| `gpt-5.4-mini` | `hipporag_budget_matched` | `8` | `0.25` |
| `gpt-5.4-mini` | `raw` | `8` | `0.25` |
| `gpt-5.4-mini` | `raw_budget_matched` | `8` | `0.375` |

## Route Credit

| model | problems | agent acc | recoverable agent errors | unrecoverable agent errors | losses to controls | VOI actions |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `gpt-5.4-mini` | `8` | `0.5` | `1` | `3` | `hipporag_baseline:1` | `continue_exploration:5, none:1, preserve_route:2` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `process_status_counts` | `completed` | `8` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `8` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `8` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `1824` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `8` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `40` |
| `cache_live_separation` | `planned_live_model_calls` | `40` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `40` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `168` |
| `context_pollution_summary` | `evidence_no_results` | `4` |
| `context_pollution_summary` | `generic_graph_context_only` | `6` |
| `context_pollution_summary` | `graph_context_correct` | `1` |
| `context_pollution_summary` | `graph_context_discarded` | `6` |
| `context_pollution_summary` | `graph_context_used` | `2` |
| `context_pollution_summary` | `graph_context_wrong` | `1` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `6` |
| `context_pollution_summary` | `graph_retrieval_activated` | `8` |
| `context_pollution_summary` | `hipporag_no_results` | `8` |
| `context_pollution_summary` | `morphism_correct` | `2` |
| `context_pollution_summary` | `morphism_hit` | `4` |
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

## Model Budget Fairness

| bucket | key | value |
| --- | --- | --- |
| `summary` | `agent_row_count` | `8` |
| `summary` | `agent_top_models` | `['gpt-5.4-mini']` |
| `summary` | `stronger_or_different_effective_models` | `[]` |
| `summary` | `multi_call_agent_row_count` | `8` |
| `summary` | `budget_target_models` | `['gpt-5.4-mini']` |
| `summary` | `missing_same_model_controls` | `[]` |
| `summary` | `missing_strong_baseline_controls` | `[]` |
| `summary` | `missing_budget_matched_controls` | `[]` |
| `fairness_gate` | `budget_matched_controls_present_if_needed` | `True` |
| `fairness_gate` | `model_budget_fairness_accounted` | `True` |
| `fairness_gate` | `model_budget_metadata_complete` | `True` |
| `fairness_gate` | `same_model_controls_present` | `True` |
| `fairness_gate` | `strong_baseline_controls_present_if_needed` | `True` |

## Selection Credit

| method | n | correct | error | accuracy |
| --- | ---: | ---: | ---: | ---: |
| `candidate_claim_verifier_priority` | `1` | `1` | `0` | `1.0` |
| `counter_assumption_verifier_choice` | `1` | `1` | `0` | `1.0` |
| `route_value_verifier_choice` | `2` | `0` | `0` | `0.0` |
| `unknown` | `16` | `6` | `0` | `0.375` |
| `verified_or_abstain_direct_fallback` | `20` | `7` | `0` | `0.35` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `4` |
| `agent_failure_buckets` | `evidence_invalid_or_unhelpful` | `1` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `4` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `2` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `2` |
| `agent_gain_loss` | `agent_correct_raw_wrong` | `2` |
| `agent_gain_loss` | `agent_only_correct` | `1` |
| `agent_gain_loss` | `all_three_wrong` | `3` |
| `agent_gain_loss` | `hipporag_also_wrong_agent_no_gain` | `3` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `4` |
| `agent_selection_methods` | `candidate_claim_verifier_priority` | `1` |
| `agent_selection_methods` | `counter_assumption_verifier_choice` | `1` |
| `agent_selection_methods` | `route_value_verifier_choice` | `2` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `4` |
| `verified_or_abstain_gate_status` | `abstained` | `4` |
| `verified_or_abstain_gate_status` | `allowed` | `4` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::correct` | `4` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `4` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::correct` | `4` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::wrong_or_error` | `4` |
| `by_variant_answer_type::hipporag_budget_matched` | `multipleChoice::correct` | `2` |
| `by_variant_answer_type::hipporag_budget_matched` | `multipleChoice::wrong_or_error` | `6` |
| `by_variant_answer_type::raw` | `multipleChoice::correct` | `2` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `6` |
| `by_variant_answer_type::raw_budget_matched` | `multipleChoice::correct` | `3` |
| `by_variant_answer_type::raw_budget_matched` | `multipleChoice::wrong_or_error` | `5` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::correct` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::correct` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `3` |
| `by_variant_domain::hipporag_baseline` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::hipporag_baseline` | `math::correct` | `2` |
| `by_variant_domain::hipporag_baseline` | `science::correct` | `2` |
| `by_variant_domain::hipporag_baseline` | `science::wrong_or_error` | `3` |
| `by_variant_domain::hipporag_budget_matched` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::hipporag_budget_matched` | `math::correct` | `2` |
| `by_variant_domain::hipporag_budget_matched` | `science::wrong_or_error` | `5` |
| `by_variant_domain::raw` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `math::correct` | `1` |
| `by_variant_domain::raw` | `math::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `science::correct` | `1` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `4` |
| `by_variant_domain::raw_budget_matched` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::raw_budget_matched` | `math::correct` | `2` |
| `by_variant_domain::raw_budget_matched` | `science::correct` | `1` |
| `by_variant_domain::raw_budget_matched` | `science::wrong_or_error` | `4` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `316.1951` | `1` | `1547` | `none` |
| `1` | `completed` | `0` | `306.0705` | `1` | `1130` | `none` |
| `2` | `completed` | `0` | `481.3619` | `1` | `655` | `none` |
| `3` | `completed` | `0` | `406.0769` | `1` | `1586` | `none` |
| `4` | `completed` | `0` | `305.7615` | `1` | `1179` | `none` |
| `5` | `completed` | `0` | `621.4018` | `1` | `681` | `none` |
| `6` | `completed` | `0` | `170.2899` | `1` | `1630` | `none` |
| `7` | `completed` | `0` | `280.4123` | `1` | `1219` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
