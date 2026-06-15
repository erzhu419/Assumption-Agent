# HLE Module Activation Audit

- pass: `True`
- old artifacts had module telemetry: `False`
- old rows without module trace: `16`
- assumption wrapper was prompt-only: `True`
- failed gates: `[]`

## Score Delta

| artifact | model | wrapper minus raw accuracy | raw accuracy | wrapper accuracy |
| --- | --- | ---: | ---: | ---: |
| `hle_text_smoke_gpt55_raw_vs_assumption_n4_20260615.json` | `gpt-5.5` | `-0.25` | `0.5` | `0.25` |
| `hle_text_smoke_gpt54mini_raw_vs_assumption_n4_20260615.json` | `gpt-5.4-mini` | `0.25` | `0.0` | `0.25` |

## Missing Expected Modules

| model/variant | missing expected modules |
| --- | --- |
| `gpt-5.4-mini::assumption_wrapper` | `assumption_graph_retrieval, multi_candidate_self_verifier, recursive_assumption_runner, residual_writeback, structural_morphism_transfer, world_model_router` |
| `gpt-5.5::assumption_wrapper` | `assumption_graph_retrieval, multi_candidate_self_verifier, recursive_assumption_runner, residual_writeback, structural_morphism_transfer, world_model_router` |

## Diagnosis

- The old `assumption_wrapper` was a single prompt scaffold, not the full Assumption Agent execution chain.
- The old logs could localize failures only to the API-call boundary; they could not show a stuck internal module.
- The updated HLE runner now emits per-call JSONL with start/end/error events, timeout seconds, latency, and module trace.

## Claim Boundary

This audit diagnoses the HLE smoke wrapper, not the full Assumption Agent.  The old HLE run used a single prompt scaffold, so equal-or-worse HLE scores should not be interpreted as a failure of graph retrieval, morphism transfer, world-model routing, or recursive self-evolution modules.
