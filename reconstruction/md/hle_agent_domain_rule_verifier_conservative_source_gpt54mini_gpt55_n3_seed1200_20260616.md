# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `3`
- scanned rows: `1221`
- live calls returned: `3/3`
- underlying model calls executed: `25`
- live attempts resolved: `3/3`
- live call errors: `0`
- overall accuracy: `1.0`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `3` | `1.0` | `1.0` | `None` |

## Same-Batch Control Comparison

| model | comparison | shared n | agent acc | control acc | delta | agent-only correct | control-only correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `none` | `answer_format_verifier, answer_type_router, assumption_graph_retrieval, counter_assumption_challenge, critic_model_router, critic_synthesis_child, domain_router, domain_rule_mc_verifier, hle_evidence_bridge, mc_option_evidence_scorer, mc_option_sweep_candidates, multi_candidate_self_verifier, option_elimination_challenge, prompt_builder, recursive_assumption_runner, recursive_child_validation, structural_morphism_transfer, world_model_router` |

## Component Efficacy

| model | variant | selection methods | key functional flags | flag accuracy |
| --- | --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `candidate_claim_verifier_priority:2, counter_assumption_verifier_choice:1` | `graph_context_injected:3, evidence_bridge_activated:3, evidence_child_executed:3, recursive_diverse_candidates:3, critic_model_used:3, claim_verifier_no_executable_claim:2, domain_rule_mc_verifier_activated:2, domain_rule_mc_verifier_selected:2` | `graph_context_injected:1.0, evidence_bridge_activated:1.0, evidence_child_executed:1.0, recursive_diverse_candidates:1.0, critic_model_used:1.0` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
