# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `3`
- scanned rows: `1725`
- live calls returned: `9/9`
- underlying model calls executed: `20`
- live attempts resolved: `9/9`
- live call errors: `0`
- overall accuracy: `0.0`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `3` | `0.0` | `None` | `0.0` |
| `gpt-5.4-mini` | `hipporag_baseline` | `3` | `0.0` | `None` | `0.0` |
| `gpt-5.4-mini` | `raw` | `3` | `0.0` | `None` | `0.0` |

## Same-Batch Control Comparison

| model | comparison | shared n | agent acc | control acc | delta | agent-only correct | control-only correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `agent_vs_hipporag_baseline` | `3` | `0.0` | `0.0` | `0.0` | `0` | `0` |
| `gpt-5.4-mini` | `agent_vs_raw` | `3` | `0.0` | `0.0` | `0.0` | `0` | `0` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `hle_evidence_bridge` | `answer_format_verifier, answer_type_router, assumption_graph_retrieval, counter_assumption_challenge, domain_router, hle_evidence_bridge, multi_candidate_self_verifier, prompt_builder, recursive_assumption_runner, recursive_child_validation, structural_morphism_transfer, world_model_router` |
| `gpt-5.4-mini` | `hipporag_baseline` | `none` | `answer_format_verifier, answer_type_router, hipporag_associative_rerank, hipporag_context_retrieval, prompt_builder` |
| `gpt-5.4-mini` | `raw` | `none` | `answer_format_verifier, answer_type_router` |

## Component Efficacy

| model | variant | selection methods | key functional flags | flag accuracy |
| --- | --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `exact_direct_fallback:1, math_exact_normalized_majority:1, normalized_majority:1` | `evidence_bridge_activated:2, evidence_child_executed:2, recursive_diverse_candidates:1, recursive_collapsed_consensus:2, recursive_timeout_pressure:1, claim_verifier_no_executable_claim:1, counter_assumption_challenge_activated:2, majority_only_selection:2` | `evidence_bridge_activated:0.0, evidence_child_executed:0.0, recursive_diverse_candidates:0.0, recursive_collapsed_consensus:0.0, recursive_timeout_pressure:0.0` |
| `gpt-5.4-mini` | `hipporag_baseline` | `none:3` | `context_injected:3` | `context_injected:0.0` |
| `gpt-5.4-mini` | `raw` | `none:3` | `none` | `none` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
