# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `1`
- scanned rows: `18`
- live calls returned: `2/2`
- underlying model calls executed: `5`
- live attempts resolved: `2/2`
- live call errors: `0`
- overall accuracy: `0.0`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` | `assumption_agent_recursive_verify` | `1` | `0.0` | `None` | `0.0` |
| `gpt-5.5` | `raw` | `1` | `0.0` | `None` | `0.0` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.5` | `assumption_agent_recursive_verify` | `none` | `answer_format_repair, answer_format_verifier, answer_type_router, assumption_graph_retrieval, domain_router, multi_candidate_self_verifier, prompt_builder, recursive_assumption_runner, recursive_child_validation, structural_morphism_transfer, world_model_router` |
| `gpt-5.5` | `raw` | `none` | `answer_format_verifier, answer_type_router` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
