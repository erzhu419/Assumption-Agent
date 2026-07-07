# HLE Fast Policy Promotion Report

- pass: `False`
- recommendation: `do_not_promote_collect_more_unseen_or_fix_blockers`
- row count: `36`
- complete triads: `12`
- promotion blockers: `['insufficient_unseen_triads_min_24', 'agent_no_fallback_present', 'control_or_agent_error_rows_present', 'no_selector_policy_gain']`

## Source Audit

| fresh exclusion | pollution pass | paper clean | planned shards | excluded old problems | duplicate sample hashes | top-level errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `True` | `True` | `False` | `12` | `6108` | `0` | `2` |

## Triad Accuracy

| model | variant | n | correct | accuracy | errors |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `12` | `2` | `0.1667` | `0` |
| `gpt-5.4-mini` | `hipporag_baseline` | `12` | `2` | `0.1667` | `1` |
| `gpt-5.4-mini` | `raw` | `12` | `2` | `0.1667` | `1` |

## Agent Vs Controls

| model | best control | agent acc | control acc | margin | passed |
| --- | --- | ---: | ---: | ---: | --- |
| `gpt-5.4-mini` | `raw` | `0.1667` | `0.1667` | `0.0` | `True` |

## Selector Simulation

| policy | correct | accuracy | delta vs agent |
| --- | ---: | ---: | ---: |
| `agent_current` | `2` | `0.1667` | `0.0` |
| `always_hipporag` | `2` | `0.1667` | `0.0` |
| `always_raw` | `2` | `0.1667` | `0.0` |
| `baseline_consensus_else_agent` | `2` | `0.1667` | `0.0` |
| `verified_else_hipporag` | `2` | `0.1667` | `0.0` |
| `verified_else_raw` | `2` | `0.1667` | `0.0` |

## Agent Failure Mining

- hypotheses: `['deterministic_option_coverage_and_required_term_source_bundle', 'candidate_specific_direct_relation_span_bundle', 'preserve_slow_baseline_when_verified_gate_has_no_direct_candidate', 'batch_or_cap_source_directness_calls_before_slow_baseline_fallback']`
- miner blockers: `['insufficient_unseen_transition_rows_min_24', 'no_fallback_present', 'missing_fair_control_or_split_metadata']`

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
