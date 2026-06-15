# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `8`
- scanned rows: `13`
- live calls returned: `16/16`
- live attempts resolved: `16/16`
- live call errors: `0`
- overall accuracy: `0.375`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` | `assumption_agent` | `8` | `0.5` | `1.0` | `0.4286` |
| `gpt-5.5` | `raw` | `8` | `0.25` | `1.0` | `0.1429` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.5` | `assumption_agent` | `none` | `answer_format_verifier, answer_type_router, assumption_graph_retrieval, domain_router, prompt_builder, recursive_assumption_runner, structural_morphism_transfer, world_model_router` |
| `gpt-5.5` | `raw` | `none` | `answer_format_verifier, answer_type_router` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
