# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `1`
- scanned rows: `115`
- live calls returned: `2/2`
- underlying model calls executed: `6`
- live attempts resolved: `2/2`
- live call errors: `0`
- overall accuracy: `1.0`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `1` | `1.0` | `1.0` | `None` |
| `gpt-5.4-mini` | `raw` | `1` | `1.0` | `1.0` | `None` |

## Same-Batch Control Comparison

| model | comparison | shared n | agent acc | control acc | delta | agent-only correct | control-only correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `agent_vs_raw` | `1` | `1.0` | `1.0` | `0.0` | `0` | `0` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `agent_hipporag_context_bridge` | `answer_format_verifier, answer_type_router, assumption_graph_retrieval, child_model_router, counter_assumption_challenge, critic_model_router, critic_synthesis_child, domain_router, mc_option_sweep_candidates, multi_candidate_self_verifier, option_elimination_challenge, prompt_builder, recursive_assumption_runner, recursive_child_validation, structural_morphism_transfer, world_model_router` |
| `gpt-5.4-mini` | `raw` | `none` | `answer_format_verifier, answer_type_router` |

## Component Efficacy

| model | variant | selection methods | key functional flags | flag accuracy |
| --- | --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_agent_recursive_verify` | `verified_or_abstain_direct_fallback:1` | `recursive_diverse_candidates:1, critic_model_used:1, critic_synthesis_activated:1, mc_option_sweep_activated:1, counter_assumption_challenge_activated:1, option_elimination_challenge_activated:1` | `recursive_diverse_candidates:1.0, critic_model_used:1.0, critic_synthesis_activated:1.0, mc_option_sweep_activated:1.0, counter_assumption_challenge_activated:1.0` |
| `gpt-5.4-mini` | `raw` | `none:1` | `none` | `none` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
