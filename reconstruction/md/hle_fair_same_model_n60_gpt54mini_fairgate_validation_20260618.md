# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `False`
- pollution pass: `True`
- parallel workers: `16`
- launch stagger sec: `0.1`
- process timeout policy: `watch_only`
- kill on soft timeout: `False`
- reused completed shards: `60`
- shard sample dedupe: `False`
- loaded shard payloads: `60/60`
- sample count: `60`
- distinct sample problems: `60`
- duplicate sample problems: `0`
- live attempts resolved: `180/180`
- scored rows: `180`
- overall accuracy: `0.1667`
- top-level live errors: `1`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `['zero_top_level_live_errors', 'budget_matched_controls_present_if_needed', 'model_budget_fairness_accounted']`
- pollution failed gates: `[]`
- model-budget fairness failed gates: `['budget_matched_controls_present_if_needed', 'model_budget_fairness_accounted']`
- recommended HLE claim scope: `clean_shared_subset_due_to_endpoint_noise`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `60` | `0.1667` | `0` | `0.1667` | `None` |
| `gpt-5.4-mini` | `hipporag_baseline` | `60` | `0.1833` | `1` | `0.1833` | `None` |
| `gpt-5.4-mini` | `raw` | `60` | `0.15` | `0` | `0.15` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `59` | `0.1695` |
| `gpt-5.4-mini` | `hipporag_baseline` | `59` | `0.1864` |
| `gpt-5.4-mini` | `raw` | `59` | `0.1525` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
| `top_level_errors_by_variant` | `hipporag_baseline` | `1` |
| `top_level_errors_by_type` | `RuntimeError` | `1` |
| `top_level_errors_by_variant_type` | `hipporag_baseline::RuntimeError` | `1` |
| `top_level_errors_by_label` | `RemoteDisconnected` | `1` |
| `top_level_errors_by_variant_label` | `hipporag_baseline::RemoteDisconnected` | `1` |
| `jsonl_error_events_by_event` | `call_error` | `1` |
| `jsonl_error_events_by_variant` | `hipporag_baseline` | `1` |
| `jsonl_error_events_by_type` | `RuntimeError` | `1` |
| `jsonl_error_events_by_label` | `RemoteDisconnected` | `1` |
| `process_status_counts` | `completed` | `60` |

## Pollution Audit

| bucket | key | value |
| --- | --- | ---: |
| `fresh_problem_hash_exclusion` | `distinct_sample_problem_hash_count` | `60` |
| `fresh_problem_hash_exclusion` | `duplicate_sample_problem_hash_count` | `0` |
| `fresh_problem_hash_exclusion` | `exclude_existing_enabled_shard_count` | `0` |
| `fresh_problem_hash_exclusion` | `excluded_existing_problem_count` | `0` |
| `fresh_problem_hash_exclusion` | `sample_problem_hash_count` | `60` |
| `cache_live_separation` | `execute_live` | `True` |
| `cache_live_separation` | `live_model_calls_executed` | `179` |
| `cache_live_separation` | `planned_live_model_calls` | `180` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `180` |
| `cache_live_separation` | `top_level_error_count` | `1` |
| `cache_live_separation` | `underlying_model_calls_executed` | `431` |
| `context_pollution_summary` | `generic_graph_context_only` | `43` |
| `context_pollution_summary` | `graph_context_correct` | `2` |
| `context_pollution_summary` | `graph_context_discarded` | `43` |
| `context_pollution_summary` | `graph_context_used` | `11` |
| `context_pollution_summary` | `graph_context_wrong` | `9` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `43` |
| `context_pollution_summary` | `graph_retrieval_activated` | `60` |
| `context_pollution_summary` | `hipporag_context_used` | `3` |
| `context_pollution_summary` | `hipporag_context_wrong` | `3` |
| `context_pollution_summary` | `hipporag_no_results` | `50` |
| `context_pollution_summary` | `morphism_correct` | `5` |
| `context_pollution_summary` | `morphism_hit` | `32` |
| `context_pollution_summary` | `morphism_wrong` | `27` |
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
| `summary` | `agent_row_count` | `60` |
| `summary` | `agent_top_models` | `['gpt-5.4-mini']` |
| `summary` | `stronger_or_different_effective_models` | `[]` |
| `summary` | `multi_call_agent_row_count` | `60` |
| `summary` | `budget_target_models` | `['gpt-5.4-mini']` |
| `summary` | `missing_same_model_controls` | `[]` |
| `summary` | `missing_strong_baseline_controls` | `[]` |
| `summary` | `missing_budget_matched_controls` | `[{'model': 'gpt-5.4-mini', 'missing_variants': ['raw_budget_matched', 'hipporag_budget_matched'], 'present_variants': ['assumption_agent_recursive_verify', 'hipporag_baseline', 'raw']}]` |
| `fairness_gate` | `budget_matched_controls_present_if_needed` | `False` |
| `fairness_gate` | `model_budget_fairness_accounted` | `False` |
| `fairness_gate` | `model_budget_metadata_complete` | `True` |
| `fairness_gate` | `same_model_controls_present` | `True` |
| `fairness_gate` | `strong_baseline_controls_present_if_needed` | `True` |

## Selection Credit

| method | n | correct | error | accuracy |
| --- | ---: | ---: | ---: | ---: |
| `candidate_claim_verifier_priority` | `4` | `2` | `0` | `0.5` |
| `unknown` | `120` | `20` | `1` | `0.1667` |
| `verified_or_abstain_direct_fallback` | `56` | `8` | `0` | `0.1429` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `50` |
| `agent_failure_buckets` | `hipporag_context_invalid_or_unhelpful` | `3` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `50` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `48` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `23` |
| `agent_failure_buckets` | `weak_morphism_unhelpful` | `4` |
| `agent_gain_loss` | `agent_correct_raw_wrong` | `4` |
| `agent_gain_loss` | `agent_only_correct` | `2` |
| `agent_gain_loss` | `all_three_wrong` | `43` |
| `agent_gain_loss` | `hipporag_also_wrong_agent_no_gain` | `45` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `47` |
| `agent_gain_loss` | `raw_correct_agent_wrong_regression` | `3` |
| `agent_selection_methods` | `candidate_claim_verifier_priority` | `4` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `56` |
| `verified_or_abstain_gate_status` | `abstained` | `56` |
| `verified_or_abstain_gate_status` | `allowed` | `4` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::correct` | `10` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `50` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::correct` | `11` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::wrong_or_error` | `49` |
| `by_variant_answer_type::raw` | `multipleChoice::correct` | `9` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `51` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::correct` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::correct` | `3` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::wrong_or_error` | `6` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::correct` | `6` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `41` |
| `by_variant_domain::assumption_agent_recursive_verify` | `software_engineering::wrong_or_error` | `1` |
| `by_variant_domain::hipporag_baseline` | `hle_general::correct` | `1` |
| `by_variant_domain::hipporag_baseline` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::hipporag_baseline` | `math::correct` | `2` |
| `by_variant_domain::hipporag_baseline` | `math::wrong_or_error` | `7` |
| `by_variant_domain::hipporag_baseline` | `science::correct` | `8` |
| `by_variant_domain::hipporag_baseline` | `science::wrong_or_error` | `39` |
| `by_variant_domain::hipporag_baseline` | `software_engineering::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `hle_general::wrong_or_error` | `3` |
| `by_variant_domain::raw` | `math::correct` | `3` |
| `by_variant_domain::raw` | `math::wrong_or_error` | `6` |
| `by_variant_domain::raw` | `science::correct` | `6` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `41` |
| `by_variant_domain::raw` | `software_engineering::wrong_or_error` | `1` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `None` | `1` | `3000` | `none` |
| `1` | `completed` | `0` | `None` | `1` | `3400` | `none` |
| `2` | `completed` | `0` | `None` | `1` | `3800` | `none` |
| `3` | `completed` | `0` | `None` | `1` | `4200` | `none` |
| `4` | `completed` | `0` | `None` | `1` | `4600` | `none` |
| `5` | `completed` | `0` | `None` | `1` | `5000` | `none` |
| `6` | `completed` | `0` | `None` | `1` | `5400` | `none` |
| `7` | `completed` | `0` | `None` | `1` | `5800` | `none` |
| `8` | `completed` | `0` | `None` | `1` | `6200` | `none` |
| `9` | `completed` | `0` | `None` | `1` | `6600` | `none` |
| `10` | `completed` | `0` | `None` | `1` | `7000` | `none` |
| `11` | `completed` | `0` | `None` | `1` | `7400` | `none` |
| `12` | `completed` | `0` | `None` | `1` | `7800` | `none` |
| `13` | `completed` | `0` | `None` | `1` | `8200` | `none` |
| `14` | `completed` | `0` | `None` | `1` | `8600` | `none` |
| `15` | `completed` | `0` | `None` | `1` | `9000` | `none` |
| `16` | `completed` | `0` | `None` | `1` | `9400` | `none` |
| `17` | `completed` | `0` | `None` | `1` | `9800` | `none` |
| `18` | `completed` | `0` | `None` | `1` | `10200` | `none` |
| `19` | `completed` | `0` | `None` | `1` | `10600` | `none` |
| `20` | `completed` | `0` | `None` | `1` | `11000` | `none` |
| `21` | `completed` | `0` | `None` | `1` | `11400` | `none` |
| `22` | `completed` | `0` | `None` | `1` | `11800` | `none` |
| `23` | `completed` | `0` | `None` | `1` | `12200` | `none` |
| `24` | `completed` | `0` | `None` | `1` | `12600` | `none` |
| `25` | `completed` | `0` | `None` | `1` | `13000` | `none` |
| `26` | `completed` | `0` | `None` | `1` | `13400` | `none` |
| `27` | `completed` | `0` | `None` | `1` | `13800` | `none` |
| `28` | `completed` | `0` | `None` | `1` | `14200` | `none` |
| `29` | `completed` | `0` | `None` | `1` | `14600` | `none` |
| `30` | `completed` | `0` | `None` | `1` | `15000` | `none` |
| `31` | `completed` | `0` | `None` | `1` | `15400` | `none` |
| `32` | `completed` | `0` | `None` | `1` | `15800` | `none` |
| `33` | `completed` | `0` | `None` | `1` | `16200` | `none` |
| `34` | `completed` | `0` | `None` | `1` | `16600` | `none` |
| `35` | `completed` | `0` | `None` | `1` | `17000` | `none` |
| `36` | `completed` | `0` | `None` | `1` | `17400` | `none` |
| `37` | `completed` | `0` | `None` | `1` | `17800` | `none` |
| `38` | `completed` | `0` | `None` | `1` | `18200` | `none` |
| `39` | `completed` | `0` | `None` | `1` | `18600` | `none` |
| `40` | `completed` | `0` | `None` | `1` | `19000` | `none` |
| `41` | `completed` | `0` | `None` | `1` | `19400` | `none` |
| `42` | `completed` | `0` | `None` | `1` | `19800` | `none` |
| `43` | `completed` | `0` | `None` | `1` | `20200` | `none` |
| `44` | `completed` | `0` | `None` | `1` | `20600` | `none` |
| `45` | `completed` | `0` | `None` | `1` | `21000` | `none` |
| `46` | `completed` | `0` | `None` | `1` | `21400` | `none` |
| `47` | `completed` | `0` | `None` | `1` | `21800` | `none` |
| `48` | `completed` | `0` | `None` | `1` | `22200` | `none` |
| `49` | `completed` | `0` | `None` | `1` | `22600` | `none` |
| `50` | `completed` | `0` | `None` | `1` | `23000` | `none` |
| `51` | `completed` | `0` | `None` | `1` | `23400` | `none` |
| `52` | `completed` | `0` | `None` | `1` | `23800` | `none` |
| `53` | `completed` | `0` | `None` | `1` | `24200` | `none` |
| `54` | `completed` | `0` | `None` | `1` | `24600` | `none` |
| `55` | `completed` | `0` | `None` | `1` | `25000` | `none` |
| `56` | `completed` | `0` | `None` | `1` | `25400` | `none` |
| `57` | `completed` | `0` | `None` | `1` | `25800` | `none` |
| `58` | `completed` | `0` | `None` | `1` | `26200` | `none` |
| `59` | `completed` | `0` | `None` | `1` | `26600` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
