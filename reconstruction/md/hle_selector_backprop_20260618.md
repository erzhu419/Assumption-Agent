# HLE Selector Backprop

- pass: `True`
- row count: `180`
- complete triads: `60`
- agent regressions where any control is correct: `7`
- failed gates: `[]`

## Policy Simulation

| policy | correct | accuracy | delta vs current |
| --- | ---: | ---: | ---: |
| `agent_current` | `10` | `0.1667` | `0.0` |
| `always_hipporag` | `11` | `0.1833` | `0.0167` |
| `always_raw` | `9` | `0.15` | `-0.0167` |
| `baseline_consensus_else_agent` | `9` | `0.15` | `-0.0167` |
| `verified_else_hipporag` | `12` | `0.2` | `0.0333` |
| `verified_else_raw` | `9` | `0.15` | `-0.0167` |

## Recommended Adjustments

| adjustment | count |
| --- | ---: |
| `prefer_hipporag_preserve_selector_for_unverified_mc` | `4` |
| `prefer_raw_preserve_selector_for_unverified_mc` | `2` |
| `tighten_candidate_claim_verifier_with_baseline_negative_control` | `1` |

## Loss Buckets

| bucket | count |
| --- | ---: |
| `hipporag_only_correct::abstained::verified_or_abstain_direct_fallback` | `3` |
| `hipporag_only_correct::allowed::candidate_claim_verifier_priority` | `1` |
| `raw_and_hipporag_correct::abstained::verified_or_abstain_direct_fallback` | `1` |
| `raw_only_correct::abstained::verified_or_abstain_direct_fallback` | `2` |

Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.
