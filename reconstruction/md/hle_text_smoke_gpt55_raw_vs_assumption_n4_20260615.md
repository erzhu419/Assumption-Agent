# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `4`
- scanned rows: `5`
- live calls returned: `5/8`
- live attempts resolved: `8/8`
- live call errors: `3`
- overall accuracy: `0.375`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.5` | `assumption_wrapper` | `4` | `0.25` | `1.0` | `0.0` |
| `gpt-5.5` | `raw` | `4` | `0.5` | `1.0` | `0.3333` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
