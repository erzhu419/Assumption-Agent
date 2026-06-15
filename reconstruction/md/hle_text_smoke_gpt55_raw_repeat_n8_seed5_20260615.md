# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `8`
- scanned rows: `13`
- live calls returned: `7/8`
- live attempts resolved: `8/8`
- live call errors: `1`
- overall accuracy: `0.375`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` | `raw_repeat` | `8` | `0.375` | `0.0` | `0.4286` |

## Module Activation

| model | variant | expected missing modules | activated modules |
| --- | --- | --- | --- |
| `gpt-5.5` | `raw_repeat` | `none` | `answer_format_verifier, answer_type_router` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
