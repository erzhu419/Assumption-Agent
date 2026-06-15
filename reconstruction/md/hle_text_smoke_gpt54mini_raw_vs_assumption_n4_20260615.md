# HLE Text-Only Smoke Evaluation

- pass: `True`
- dataset accessible: `True`
- sample count: `4`
- scanned rows: `5`
- live calls returned: `7/8`
- live attempts resolved: `8/8`
- live call errors: `1`
- overall accuracy: `0.125`
- raw content persisted: `False`
- failed gates: `[]`

## By Variant

| model | variant | n | accuracy | MCQ accuracy | exact accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.4-mini` | `assumption_wrapper` | `4` | `0.25` | `0.0` | `0.3333` |
| `gpt-5.4-mini` | `raw` | `4` | `0.0` | `0.0` | `0.0` |

## Claim Boundary

This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset statistics, and official leaderboard claims are out of scope.

The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.
