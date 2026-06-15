# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `1`
- scanned rows: `2`
- live calls returned: `2/2`
- live attempts resolved: `2/2`
- live call errors: `0`
- overall accuracy: `0.0`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_wrapper` | `1` | `0.0` | `0.0` | `None` |
| `gpt-5.4-mini` | `raw` | `1` | `0.0` | `0.0` | `None` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | `assumption_wrapper` | `assumption_graph_retrieval, multi_candidate_self_verifier, recursive_assumption_runner, residual_writeback, structural_morphism_transfer, world_model_router` | `answer_format_verifier, answer_type_router, prompt_scaffold` |
| `gpt-5.4-mini` | `raw` | `none` | `answer_format_verifier, answer_type_router` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
