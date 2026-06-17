# HLE Parallel Shard Evaluation

- pass: `True`
- paper clean pass: `True`
- pollution pass: `True`
- process timeout policy: `watch_only`
- kill on soft timeout: `False`
- shard sample dedupe: `completed`
- loaded shard payloads: `60/60`
- sample count: `60`
- distinct sample problems: `60`
- duplicate sample problems: `0`
- live attempts resolved: `180/180`
- scored rows: `180`
- overall accuracy: `0.2333`
- top-level live errors: `0`
- process timeouts: `0`
- failed gates: `[]`
- paper-clean failed gates: `[]`
- pollution failed gates: `[]`
- recommended HLE claim scope: `full_resolved_rows`

## By Variant

| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `60` | `0.4` | `0` | `0.4` | `None` |
| `gpt-5.4-mini` | `hipporag_baseline` | `60` | `0.1667` | `0` | `0.1667` | `None` |
| `gpt-5.4-mini` | `raw` | `60` | `0.1333` | `0` | `0.1333` | `None` |

## Clean Shared Subset

| model | variant | clean shared n | accuracy |
| --- | --- | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `60` | `0.4` |
| `gpt-5.4-mini` | `hipporag_baseline` | `60` | `0.1667` |
| `gpt-5.4-mini` | `raw` | `60` | `0.1333` |

## Error Stratification

| bucket | key | count |
| --- | --- | ---: |
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
| `cache_live_separation` | `live_model_calls_executed` | `180` |
| `cache_live_separation` | `planned_live_model_calls` | `180` |
| `cache_live_separation` | `process_timeout_count` | `0` |
| `cache_live_separation` | `resolved_live_model_calls` | `180` |
| `cache_live_separation` | `top_level_error_count` | `0` |
| `cache_live_separation` | `underlying_model_calls_executed` | `477` |
| `context_pollution_summary` | `generic_graph_context_only` | `43` |
| `context_pollution_summary` | `graph_context_correct` | `5` |
| `context_pollution_summary` | `graph_context_discarded` | `43` |
| `context_pollution_summary` | `graph_context_used` | `11` |
| `context_pollution_summary` | `graph_context_wrong` | `6` |
| `context_pollution_summary` | `graph_generic_harness_retrieved` | `43` |
| `context_pollution_summary` | `graph_retrieval_activated` | `60` |
| `context_pollution_summary` | `hipporag_context_correct` | `3` |
| `context_pollution_summary` | `hipporag_context_used` | `12` |
| `context_pollution_summary` | `hipporag_context_wrong` | `9` |
| `context_pollution_summary` | `hipporag_no_results` | `6` |
| `context_pollution_summary` | `morphism_correct` | `11` |
| `context_pollution_summary` | `morphism_hit` | `32` |
| `context_pollution_summary` | `morphism_wrong` | `21` |
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
| `candidate_claim_verifier_priority` | `4` | `3` | `0` | `0.75` |
| `unknown` | `120` | `18` | `0` | `0.15` |
| `verified_or_abstain_direct_fallback` | `56` | `21` | `0` | `0.375` |

## Failure Diagnostics

| bucket | key | count |
| --- | --- | ---: |
| `agent_failure_buckets` | `agent_wrong_or_error` | `36` |
| `agent_failure_buckets` | `hipporag_context_invalid_or_unhelpful` | `9` |
| `agent_failure_buckets` | `multiple_choice_selection_failed` | `36` |
| `agent_failure_buckets` | `verified_or_abstain_fallback_wrong` | `35` |
| `agent_failure_buckets` | `weak_morphism_routing_only_not_credited` | `18` |
| `agent_failure_buckets` | `weak_morphism_unhelpful` | `3` |
| `agent_gain_loss` | `agent_correct_raw_wrong` | `18` |
| `agent_gain_loss` | `agent_only_correct` | `15` |
| `agent_gain_loss` | `all_three_wrong` | `33` |
| `agent_gain_loss` | `hipporag_also_wrong_agent_no_gain` | `33` |
| `agent_gain_loss` | `raw_also_wrong_agent_no_gain` | `34` |
| `agent_gain_loss` | `raw_correct_agent_wrong_regression` | `2` |
| `agent_selection_methods` | `candidate_claim_verifier_priority` | `4` |
| `agent_selection_methods` | `verified_or_abstain_direct_fallback` | `56` |
| `verified_or_abstain_gate_status` | `abstained` | `56` |
| `verified_or_abstain_gate_status` | `allowed` | `4` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::correct` | `24` |
| `by_variant_answer_type::assumption_agent_recursive_verify` | `multipleChoice::wrong_or_error` | `36` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::correct` | `10` |
| `by_variant_answer_type::hipporag_baseline` | `multipleChoice::wrong_or_error` | `50` |
| `by_variant_answer_type::raw` | `multipleChoice::correct` | `8` |
| `by_variant_answer_type::raw` | `multipleChoice::wrong_or_error` | `52` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::correct` | `2` |
| `by_variant_domain::assumption_agent_recursive_verify` | `hle_general::wrong_or_error` | `1` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::correct` | `6` |
| `by_variant_domain::assumption_agent_recursive_verify` | `math::wrong_or_error` | `3` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::correct` | `15` |
| `by_variant_domain::assumption_agent_recursive_verify` | `science::wrong_or_error` | `32` |
| `by_variant_domain::assumption_agent_recursive_verify` | `software_engineering::correct` | `1` |
| `by_variant_domain::hipporag_baseline` | `hle_general::correct` | `1` |
| `by_variant_domain::hipporag_baseline` | `hle_general::wrong_or_error` | `2` |
| `by_variant_domain::hipporag_baseline` | `math::correct` | `3` |
| `by_variant_domain::hipporag_baseline` | `math::wrong_or_error` | `6` |
| `by_variant_domain::hipporag_baseline` | `science::correct` | `6` |
| `by_variant_domain::hipporag_baseline` | `science::wrong_or_error` | `41` |
| `by_variant_domain::hipporag_baseline` | `software_engineering::wrong_or_error` | `1` |
| `by_variant_domain::raw` | `hle_general::wrong_or_error` | `3` |
| `by_variant_domain::raw` | `math::correct` | `2` |
| `by_variant_domain::raw` | `math::wrong_or_error` | `7` |
| `by_variant_domain::raw` | `science::correct` | `6` |
| `by_variant_domain::raw` | `science::wrong_or_error` | `41` |
| `by_variant_domain::raw` | `software_engineering::wrong_or_error` | `1` |

## Shards

| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `0` | `completed` | `0` | `326.2421` | `1` | `0` | `none` |
| `1` | `completed` | `0` | `341.3466` | `1` | `1` | `none` |
| `2` | `completed` | `0` | `215.4457` | `1` | `2` | `none` |
| `3` | `completed` | `0` | `216.6581` | `1` | `3` | `none` |
| `4` | `completed` | `0` | `216.2615` | `1` | `4` | `none` |
| `5` | `completed` | `0` | `135.9099` | `1` | `5` | `none` |
| `6` | `completed` | `0` | `211.0656` | `1` | `6` | `none` |
| `7` | `completed` | `0` | `80.2141` | `1` | `7` | `none` |
| `8` | `completed` | `0` | `211.3357` | `1` | `8` | `none` |
| `9` | `completed` | `0` | `442.0617` | `1` | `9` | `none` |
| `10` | `completed` | `0` | `155.8496` | `1` | `10` | `none` |
| `11` | `completed` | `0` | `581.7502` | `1` | `11` | `none` |
| `12` | `completed` | `0` | `155.4632` | `1` | `12` | `none` |
| `13` | `completed` | `0` | `70.2148` | `1` | `13` | `none` |
| `14` | `completed` | `0` | `135.5422` | `1` | `14` | `none` |
| `15` | `completed` | `0` | `722.2685` | `1` | `15` | `none` |
| `16` | `completed` | `0` | `446.26` | `1` | `16` | `none` |
| `17` | `completed` | `0` | `140.4625` | `1` | `17` | `none` |
| `18` | `completed` | `0` | `205.6131` | `1` | `18` | `none` |
| `19` | `completed` | `0` | `115.354` | `1` | `19` | `none` |
| `20` | `completed` | `0` | `150.6541` | `1` | `20` | `none` |
| `21` | `completed` | `0` | `65.3332` | `1` | `21` | `none` |
| `22` | `completed` | `0` | `75.2005` | `1` | `22` | `none` |
| `23` | `completed` | `0` | `90.2435` | `1` | `23` | `none` |
| `24` | `completed` | `0` | `150.415` | `1` | `24` | `none` |
| `25` | `completed` | `0` | `55.1933` | `1` | `25` | `none` |
| `26` | `completed` | `0` | `300.8063` | `1` | `26` | `none` |
| `27` | `completed` | `0` | `170.3885` | `1` | `27` | `none` |
| `28` | `completed` | `0` | `65.1514` | `1` | `28` | `none` |
| `29` | `completed` | `0` | `110.2768` | `1` | `29` | `none` |
| `30` | `completed` | `0` | `85.2247` | `1` | `30` | `none` |
| `31` | `completed` | `0` | `255.8546` | `1` | `31` | `none` |
| `32` | `completed` | `0` | `476.3465` | `1` | `32` | `none` |
| `33` | `completed` | `0` | `937.2236` | `1` | `33` | `none` |
| `34` | `completed` | `0` | `310.6484` | `1` | `34` | `none` |
| `35` | `completed` | `0` | `110.1595` | `1` | `35` | `none` |
| `36` | `completed` | `0` | `190.3273` | `1` | `36` | `none` |
| `37` | `completed` | `0` | `135.2253` | `1` | `37` | `none` |
| `38` | `completed` | `0` | `290.7238` | `1` | `38` | `none` |
| `39` | `completed` | `0` | `130.301` | `1` | `39` | `none` |
| `40` | `completed` | `0` | `350.8312` | `1` | `40` | `none` |
| `41` | `completed` | `0` | `215.4723` | `1` | `41` | `none` |
| `42` | `completed` | `0` | `150.3198` | `1` | `42` | `none` |
| `43` | `completed` | `0` | `391.0157` | `1` | `43` | `none` |
| `44` | `completed` | `0` | `310.7501` | `1` | `44` | `none` |
| `45` | `completed` | `0` | `255.6568` | `1` | `45` | `none` |
| `46` | `completed` | `0` | `305.9136` | `1` | `46` | `none` |
| `47` | `completed` | `0` | `150.4641` | `1` | `47` | `none` |
| `48` | `completed` | `0` | `230.7015` | `1` | `48` | `none` |
| `49` | `completed` | `0` | `190.5332` | `1` | `49` | `none` |
| `50` | `completed` | `0` | `65.1999` | `1` | `50` | `none` |
| `51` | `completed` | `0` | `270.7507` | `1` | `51` | `none` |
| `52` | `completed` | `0` | `155.4323` | `1` | `52` | `none` |
| `53` | `completed` | `0` | `120.3439` | `1` | `53` | `none` |
| `54` | `completed` | `0` | `170.446` | `1` | `54` | `none` |
| `55` | `completed` | `0` | `85.1884` | `1` | `55` | `none` |
| `56` | `completed` | `0` | `95.2981` | `1` | `56` | `none` |
| `57` | `completed` | `0` | `476.3893` | `1` | `57` | `none` |
| `58` | `completed` | `0` | `917.8759` | `1` | `58` | `none` |
| `59` | `completed` | `0` | `466.3726` | `1` | `59` | `none` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
